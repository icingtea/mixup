"""The node runtime: one process that can be a leader (generate) or an expert (serve).

Holds the role-dependent state (expert group, dense model, dispatcher) and runs the
control loop: leaders generate, followers run a watchdog that promotes them on leader
failure (Phases 3/4). The FastAPI app (built in infer.py) reads this state.
"""

import gc
import logging
import threading
import time

import requests
import torch
from requests.exceptions import RequestException
from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM, LogitsProcessor

from mixup_core.constants import CHECKPOINT_FILE, MODEL_DIR, MODEL_FILE, address
from mixup_core.election import (
    apply_leader_announcement,
    current_leader,
    promote,
    should_become_leader,
)
from mixup_core.env import NodeEnv
from mixup_core.gossip import Alive, Dead, apply_gossip
from mixup_core.kvledger import Checkpoint, decode_checkpoint, encode_checkpoint
from mixup_core.runtime import ClusterView, RoutingTable
from mixup_core.scheduler import (
    LeastLoadedPolicy,
    balance_experts,
    handle_node_failure,
    rebalance_experts,
)
from mixup_core.types import MemberStatus, NodeRole

from .dispatch import Dispatcher
from .metrics import MetricsCollector
from .model import ExpertGroup, load_central

logger = logging.getLogger(__name__)


class _CheckpointProcessor(LogitsProcessor):
    """Writes the token-sequence checkpoint after each generated token (before sampling)."""

    def __init__(self, prompt_len: int, max_new_tokens: int, path) -> None:
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.path = path

    def __call__(self, input_ids, scores):
        kv_seq = input_ids.shape[1] - self.prompt_len
        checkpoint = Checkpoint(
            kv_seq=kv_seq,
            input_ids=input_ids[0].tolist(),
            max_new_tokens=self.max_new_tokens,
        )
        self.path.write_text(encode_checkpoint(checkpoint).decode())
        return scores


def broadcast_routing(view: ClusterView, by_node: str) -> None:
    """Push the current routing table to every alive expert node (so they load new experts).

    Retries each node a few times: a rejoining node's server may not be listening yet, so
    a single fire-and-forget push races its boot.
    """
    table = view.routing
    if table is None:
        return
    payload = table.model_dump()
    for member in view.members.values():
        if member.role is NodeRole.EXPERT and member.status is MemberStatus.ALIVE:
            sent = False
            for _ in range(5):
                try:
                    requests.post(
                        f"{address(member.node_id)}/routing", json=payload, timeout=60
                    )
                    logger.info(
                        "broadcast: sent routing epoch %d to %s", table.epoch, member.node_id
                    )
                    sent = True
                    break
                except RequestException:
                    time.sleep(1)
            if not sent:
                logger.warning("broadcast: could not reach %s during rebalance", member.node_id)


class Node:
    """A node's runtime state and control loop."""

    def __init__(self, env: NodeEnv, model_config: GraniteMoeConfig, view: ClusterView) -> None:
        self.env = env
        self.name = env.container_name
        self.model_config = model_config
        self.view = view
        self.role: NodeRole = env.config.initial_role(self.name)
        self.ready = False
        self.expert_group: ExpertGroup | None = None
        self.model: GraniteMoeForCausalLM | None = None
        self.dispatcher: Dispatcher | None = None
        self.tokenizer = None
        self._lock = threading.Lock()

    # -- control loop ------------------------------------------------------

    def run(self) -> None:
        """Entry point for the background control thread.

        A leader only loads the dense model and then serves ``POST /generate`` — it does
        not generate on its own. Generation starts only when a request is dispatched.
        """
        self._discover_topology()
        if self.role is NodeRole.LEADER:
            self._become_leader()
        else:
            self._load_experts()
            self._declare_rejoin()
            self._watchdog()

    def _discover_topology(self, grace_seconds: float = 5.0) -> None:
        """Learn the current leader by probing peers, demoting self if the cluster moved on.

        Polls for ``grace_seconds`` (or until a higher-epoch leader is seen) so a fast
        restart — which can race a still-settling election — still catches the
        newly-elected leader instead of re-assuming the startup role and re-loading dense.
        """
        deadline = time.monotonic() + grace_seconds
        best_epoch = 0
        best_leader: str | None = None
        while time.monotonic() < deadline:
            for name in self.view.members:
                if name == self.name:
                    continue
                try:
                    projection = requests.get(f"{address(name)}/leader", timeout=0.5).json()
                except (RequestException, ValueError):
                    continue
                leader = projection.get("leader")
                epoch = projection.get("epoch", 0)
                if leader is not None and leader != self.name and epoch > best_epoch:
                    best_epoch = epoch
                    best_leader = leader
            if best_leader is not None:
                break
            time.sleep(0.5)

        if best_leader is not None:
            logger.info(
                "%s: existing leader %s (epoch %d); joining as follower",
                self.name,
                best_leader,
                best_epoch,
            )
            with self._lock:
                self.view = apply_leader_announcement(self.view, best_leader, best_epoch)
                me = self.view.members.get(self.name)
                if me is not None:
                    self.role = me.role

    # -- role transitions --------------------------------------------------

    def _become_leader(self) -> None:
        """Shed our experts (if any), load the dense model, and become the leader."""
        if self.expert_group is not None:
            with self._lock:
                self.view = promote(self.view, self.name)
                if self.view.routing is not None:
                    new_table = rebalance_experts(
                        self.view, self.view.routing, LeastLoadedPolicy()
                    )
                    self.view = ClusterView(members=self.view.members, routing=new_table)
                shed_view = self.view
            broadcast_routing(shed_view, self.name)
            self.expert_group = None
            # Free the shed experts' GPU memory before loading the dense model, otherwise
            # the promotion can OOM (CUBLAS_STATUS_ALLOC_FAILED) on a shared GPU.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("%s: shed experts, becoming leader", self.name)

        assert self.view.routing is not None
        self.dispatcher = Dispatcher(self.name, self.view.routing, self._on_expert_failure)
        self.model = load_central(self.model_config, MODEL_FILE, self.dispatcher)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.role = NodeRole.LEADER
        self.ready = True
        logger.info("%s: dense model loaded; I am leader", self.name)
        self._announce_leader()

    def _load_experts(self) -> None:
        assert self.view.routing is not None
        self.expert_group = ExpertGroup(self.name, self.model_config, MODEL_FILE, self.view.routing)
        self.ready = True
        logger.info("%s: %d experts ready", self.name, len(self.expert_group.experts))

    def _on_expert_failure(self, node: str) -> RoutingTable:
        with self._lock:
            self.view = handle_node_failure(self.view, node, by_node=self.name)
            new_table = self.view.routing
        if new_table is not None:
            broadcast_routing(self.view, self.name)
        assert new_table is not None
        return new_table

    def accept_leader(self, node_id: str, epoch: int) -> None:
        """Adopt an announced leader (from another node's POST /leader)."""
        with self._lock:
            self.view = apply_leader_announcement(self.view, node_id, epoch)
            me = self.view.members.get(self.name)
            if me is not None:
                self.role = me.role

    def _declare_rejoin(self) -> None:
        """Tell the leader we're back, so it rebalances and hands us a share of experts."""
        leader = current_leader(self.view)
        if leader is None or leader == self.name:
            return
        try:
            requests.post(f"{address(leader)}/join", json={"node_id": self.name}, timeout=5)
            logger.info("%s: declared rejoin to leader %s", self.name, leader)
        except RequestException:
            logger.warning("%s: could not declare rejoin to %s", self.name, leader)

    def rejoin(self, node_id: str) -> None:
        """A node announced it rejoined: mark it alive and rebalance to include it."""
        with self._lock:
            member = self.view.members.get(node_id)
            if member is None:
                return
            self.view = apply_gossip(
                self.view,
                Alive(
                    node_id=node_id,
                    incarnation=member.incarnation + 1,
                    epoch=member.epoch,
                ),
            )
            if self.view.routing is not None:
                new_table = balance_experts(
                    self.view, self.view.routing, LeastLoadedPolicy()
                )
                self.view = ClusterView(members=self.view.members, routing=new_table)
                if self.dispatcher is not None:
                    self.dispatcher.set_table(new_table)
        broadcast_routing(self.view, self.name)
        logger.info("%s: rebalanced after %s rejoined", self.name, node_id)

    def _announce_leader(self) -> None:
        """Tell every other node we are the leader, so they poll us (and can fail over again)."""
        with self._lock:
            me = self.view.members[self.name]
            epoch = me.epoch
        payload = {"node_id": self.name, "epoch": epoch}
        for name in self.view.members:
            if name == self.name:
                continue
            try:
                requests.post(f"{address(name)}/leader", json=payload, timeout=5)
            except RequestException:
                logger.warning("could not announce leader to %s", name)

    # -- generation --------------------------------------------------------

    def generate(
        self, prompt: str, max_new_tokens: int
    ) -> tuple[str, int, float, float, float]:
        """Run a generation on demand (``POST /generate``).

        Returns ``(text, tokens, elapsed_seconds, dense_seconds, dispatch_seconds)``.
        """
        return self._generate(prompt, max_new_tokens)

    def _generate(
        self, prompt: str, max_new_tokens: int, resume: bool = False
    ) -> tuple[str, int, float, float, float]:
        assert self.model is not None and self.tokenizer is not None
        assert self.dispatcher is not None
        logger.info("request: prompt=%r max_new_tokens=%d", prompt, max_new_tokens)
        self.dispatcher.wait_for_experts()

        if resume and CHECKPOINT_FILE.exists():
            checkpoint = decode_checkpoint(CHECKPOINT_FILE.read_bytes())
            ids = torch.tensor([checkpoint.input_ids])
            prompt_len = len(checkpoint.input_ids) - checkpoint.kv_seq
            target = checkpoint.max_new_tokens
            remaining = max(0, target - checkpoint.kv_seq)
            logger.info(
                "resuming from checkpoint: kv_seq=%d, %d tokens left",
                checkpoint.kv_seq,
                remaining,
            )
        else:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = ids.shape[1]
            target = max_new_tokens
            remaining = max_new_tokens

        if remaining <= 0:
            logger.info("generation already complete")
            return "", 0, 0.0, 0.0, 0.0

        metrics = MetricsCollector(self.name, prompt, target)
        self.dispatcher.metrics = metrics

        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        processor = _CheckpointProcessor(prompt_len, target, CHECKPOINT_FILE)
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=remaining, do_sample=False, logits_processor=[processor]
            )
        elapsed = time.perf_counter() - start
        generated = out.shape[1] - ids.shape[1]
        throughput = generated / elapsed if elapsed > 0 else 0.0
        dense = max(0.0, elapsed - metrics.dispatch_seconds)
        dispatch = metrics.dispatch_seconds

        metrics.record_generation(tokens=generated, total=elapsed, tok_per_sec=throughput)
        metrics.close()
        self.dispatcher.metrics = None

        logger.info(
            "benchmark: generated_tokens=%d generation_seconds=%.3f dense_seconds=%.3f "
            "dispatch_seconds=%.3f tokens_per_second=%.2f",
            generated,
            elapsed,
            dense,
            dispatch,
            throughput,
        )
        text = self.tokenizer.decode(out[0][prompt_len:])
        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error("generation output contains NaN/inf — model state corrupted")
        logger.info("generated: %s", text)
        return text, generated, elapsed, dense, dispatch

    # -- watchdog ----------------------------------------------------------

    def _watchdog(self) -> None:
        """Poll the leader's /health; on failure, promote ourselves if we win the election."""
        interval = 1.0
        while True:
            time.sleep(interval)
            with self._lock:
                leader = current_leader(self.view)
                if leader is None or leader == self.name:
                    continue
                member = self.view.members.get(leader)
                incarnation = member.incarnation if member else 1

            try:
                requests.get(f"{address(leader)}/health", timeout=2)
                continue
            except RequestException:
                logger.warning("leader %s appears down; checking election", leader)

            with self._lock:
                self.view = apply_gossip(
                    self.view,
                    Dead(node_id=leader, incarnation=incarnation, by_node=self.name),
                )
                become = should_become_leader(self.view, self.name)

            if become:
                logger.info("%s: I win the election; promoting to leader", self.name)
                self.role = NodeRole.LEADER
                self._become_leader()
                # Don't generate here: the new leader waits for the next /generate request.
                return
