"""The node runtime: one process that can be a leader (generate) or an expert (serve).

Holds the role-dependent state (expert group, dense model, dispatcher) and runs the
control loop: leaders generate, followers run a watchdog that promotes them on leader
failure (Phases 3/4). The FastAPI app (built in infer.py) reads this state.
"""

import base64
import contextlib
import gc
import logging
import threading
import time

import requests
import torch
from requests.exceptions import RequestException
from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM

from mixup_core.constants import MODEL_DIR, MODEL_FILE, address
from mixup_core.election import (
    apply_leader_announcement,
    current_leader,
    promote,
    should_become_leader,
)
from mixup_core.env import NodeEnv
from mixup_core.gossip import Alive, apply_gossip, decode_gossip, encode_gossip
from mixup_core.gossip_engine import GossipEngine
from mixup_core.kvledger import KvState, decode_kv_state, encode_kv_state
from mixup_core.runtime import ClusterView, RoutingTable
from mixup_core.scheduler import (
    LeastLoadedPolicy,
    balance_experts,
    handle_node_failure,
    rebalance_experts,
)
from mixup_core.types import NodeRole

from .dispatch import Dispatcher
from .gossip_loop import GossipLoop
from .kv_cache import cache_from_bytes, cache_to_bytes
from .metrics import MetricsCollector
from .model import ExpertGroup, load_central

logger = logging.getLogger(__name__)

# SWIM gossip timing. Kept short so a node failure is detected in a few seconds (the
# trade-off is more probe traffic, which is negligible for this cluster size).
GOSSIP_INTERVAL = 0.5
PING_TIMEOUT = 1.0
SUSPECT_TIMEOUT = 2.0
INDIRECT_K = 2


def broadcast_routing(view: ClusterView, by_node: str) -> None:
    """Push the current routing table to every expert node (so they load new experts).

    Iterates the routing table's expert nodes (not the membership roster), so a node that
    was rebalanced away or died is not contacted. Retries each node a few times: a
    rejoining node's server may not be listening yet, so a single fire-and-forget push
    races its boot.
    """
    table = view.routing
    if table is None:
        return
    payload = table.model_dump()
    nodes = sorted({entry.node_id for entry in table.assignment.values()})
    for node_id in nodes:
        sent = False
        for _ in range(5):
            try:
                requests.post(f"{address(node_id)}/routing", json=payload, timeout=5)
                logger.info("broadcast: sent routing epoch %d to %s", table.epoch, node_id)
                sent = True
                break
            except RequestException:
                time.sleep(1)
        if not sent:
            logger.warning("broadcast: could not reach %s during rebalance", node_id)


class Node:
    """A node's runtime state and control loop."""

    def __init__(self, env: NodeEnv, model_config: GraniteMoeConfig, view: ClusterView) -> None:
        self.env = env
        self.name = env.container_name
        self.model_config = model_config
        self.role: NodeRole = env.config.initial_role(self.name)
        self.ready = False
        self.expert_group: ExpertGroup | None = None
        self.model: GraniteMoeForCausalLM | None = None
        self.dispatcher: Dispatcher | None = None
        self.tokenizer = None
        # The gossip engine owns the live view (single source of truth, thread-safe) and
        # the loop drives SWIM failure detection over the network.
        self.engine = GossipEngine(
            self.name,
            view,
            ping_timeout=PING_TIMEOUT,
            suspect_timeout=SUSPECT_TIMEOUT,
            indirect_k=INDIRECT_K,
        )
        self.loop = GossipLoop(
            self.engine,
            gossip_interval=GOSSIP_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            suspect_timeout=SUSPECT_TIMEOUT,
            indirect_k=INDIRECT_K,
            send=self._send_gossip,
            on_dead=self._on_member_dead,
        )
        # The freshest warm KV-cache snapshot this node has received (as a follower), for
        # resuming an interrupted generation after a failover.
        self._kv: KvState | None = None
        self._kv_seq = 0  # global monotonic token count (election freshness + dedup)
        self.kv_sync_every = env.config.kv_sync_every

    # -- control loop ------------------------------------------------------

    def run(self) -> None:
        """Entry point for the background control thread.

        Boots the node and starts the SWIM gossip loop; failure detection (leader or
        expert) is driven by the loop's ``on_dead`` callback, not by polling. The loop
        starts after the discovery grace so peers' servers are up before probes begin.
        """
        self._discover_topology()
        threading.Thread(target=self.loop.run, args=(0.1,), daemon=True).start()
        if self.role is NodeRole.LEADER:
            self._become_leader()
        else:
            self._load_experts()
            self._declare_rejoin()

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
            for name in self.engine.view.members:
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
            new_view = self.engine.mutate(
                lambda v: apply_leader_announcement(v, best_leader, best_epoch)
            )
            me = new_view.members.get(self.name)
            if me is not None:
                self.role = me.role

    # -- role transitions --------------------------------------------------

    def _become_leader(self) -> None:
        """Shed our experts (if any), load the dense model, and become the leader."""
        if self.expert_group is not None:

            def _promote(v: ClusterView) -> ClusterView:
                v = promote(v, self.name)
                if v.routing is not None:
                    new_table = rebalance_experts(v, v.routing, LeastLoadedPolicy())
                    v = ClusterView(members=v.members, routing=new_table)
                return v

            shed_view = self.engine.mutate(_promote)
            broadcast_routing(shed_view, self.name)
            self.expert_group = None
            # Free the shed experts' GPU memory before loading the dense model, otherwise
            # the promotion can OOM (CUBLAS_STATUS_ALLOC_FAILED) on a shared GPU.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("%s: shed experts, becoming leader", self.name)

        view = self.engine.view
        assert view.routing is not None
        self.dispatcher = Dispatcher(self.name, view.routing, self._on_expert_failure)
        self.model = load_central(self.model_config, MODEL_FILE, self.dispatcher)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.role = NodeRole.LEADER
        self.ready = True
        logger.info("%s: dense model loaded; I am leader", self.name)
        self._announce_leader()

    def _load_experts(self) -> None:
        view = self.engine.view
        assert view.routing is not None
        self.expert_group = ExpertGroup(self.name, self.model_config, MODEL_FILE, view.routing)
        self.ready = True
        logger.info("%s: %d experts ready", self.name, len(self.expert_group.experts))

    def _on_expert_failure(self, node: str) -> RoutingTable:
        view = self.engine.mutate(lambda v: handle_node_failure(v, node, by_node=self.name))
        new_table = view.routing
        if new_table is not None:
            broadcast_routing(view, self.name)
        assert new_table is not None
        return new_table

    def accept_leader(self, node_id: str, epoch: int) -> None:
        """Adopt an announced leader (from another node's POST /leader)."""
        new_view = self.engine.mutate(lambda v: apply_leader_announcement(v, node_id, epoch))
        me = new_view.members.get(self.name)
        if me is not None:
            self.role = me.role

    def _declare_rejoin(self) -> None:
        """Tell the leader we're back, so it rebalances and hands us a share of experts."""
        leader = current_leader(self.engine.view)
        if leader is None or leader == self.name:
            return
        try:
            requests.post(f"{address(leader)}/join", json={"node_id": self.name}, timeout=5)
            logger.info("%s: declared rejoin to leader %s", self.name, leader)
        except RequestException:
            logger.warning("%s: could not declare rejoin to %s", self.name, leader)

    def rejoin(self, node_id: str) -> None:
        """A node announced it rejoined: mark it alive and rebalance to include it."""
        member = self.engine.view.members.get(node_id)
        if member is None:
            return

        def _rebalance(v: ClusterView) -> ClusterView:
            v = apply_gossip(
                v,
                Alive(node_id=node_id, incarnation=member.incarnation + 1, epoch=member.epoch),
            )
            if v.routing is not None:
                new_table = balance_experts(v, v.routing, LeastLoadedPolicy())
                if self.dispatcher is not None:
                    self.dispatcher.set_table(new_table)
                return ClusterView(members=v.members, routing=new_table)
            return v

        new_view = self.engine.mutate(_rebalance)
        broadcast_routing(new_view, self.name)
        logger.info("%s: rebalanced after %s rejoined", self.name, node_id)

    def _announce_leader(self) -> None:
        """Tell every other node we are the leader, so they poll us (and can fail over again)."""
        me = self.engine.view.members[self.name]
        epoch = me.epoch
        payload = {"node_id": self.name, "epoch": epoch}
        for name in self.engine.view.members:
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
        self, prompt: str, max_new_tokens: int
    ) -> tuple[str, int, float, float, float]:
        assert self.model is not None and self.tokenizer is not None
        assert self.dispatcher is not None
        logger.info("request: prompt=%r max_new_tokens=%d", prompt, max_new_tokens)
        self.dispatcher.wait_for_experts()

        device = self.model.device
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        prompt_list = prompt_ids[0].tolist()

        # Resume from a warm cache if this request continues the generation we were
        # snapshotting (the gateway re-issues the same prompt after a failover).
        state = self._kv
        warm = (
            state is not None
            and len(state.input_ids) > prompt_len
            and state.input_ids[:prompt_len] == prompt_list
            and state.max_new_tokens == max_new_tokens
        )

        if warm:
            past = cache_from_bytes(base64.b64decode(state.cache_b64), device)
            nxt = torch.tensor([[state.next_token]], device=device)
            generated_ids: list[int] = state.input_ids[prompt_len:]
            remaining = max(0, max_new_tokens - len(generated_ids))
            logger.info(
                "resuming warm cache: %d tokens already generated, %d left",
                len(generated_ids),
                remaining,
            )
        else:
            past = None
            nxt = None
            generated_ids = []
            remaining = max_new_tokens

        if remaining <= 0:
            logger.info("generation already complete")
            return "", 0, 0.0, 0.0, 0.0

        metrics = MetricsCollector(self.name, prompt, max_new_tokens)
        self.dispatcher.metrics = metrics

        eos = self.tokenizer.eos_token_id
        start = time.perf_counter()
        with torch.no_grad():
            if past is None:
                out = self.model(input_ids=prompt_ids, cache=None, use_cache=True)
                past = out.cache
                nxt = out.logits[:, -1, :].argmax(dim=-1)
            for _ in range(remaining):
                if nxt.item() == eos:
                    break
                if len(generated_ids) % self.kv_sync_every == 0:
                    self._broadcast_kv(
                        past,
                        nxt.item(),
                        prompt_list + generated_ids,
                        max_new_tokens,
                    )
                generated_ids.append(nxt.item())
                self._kv_seq += 1
                self._bump_kv_seq(self._kv_seq)
                out = self.model(input_ids=nxt.unsqueeze(0), cache=past, use_cache=True)
                past = out.cache
                nxt = out.logits[:, -1, :].argmax(dim=-1)

        elapsed = time.perf_counter() - start
        generated = len(generated_ids)
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
        text = self.tokenizer.decode(generated_ids)
        logger.info("generated: %s", text)
        return text, generated, elapsed, dense, dispatch

    # -- warm KV-cache sharing --------------------------------------------

    def _broadcast_kv(
        self,
        past,
        next_token: int,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> None:
        """Snapshot the KV cache and push it to peers so a successor can resume warm.

        The cache is serialized synchronously (capturing the exact bytes at this point,
        before the loop mutates ``past`` again); the network push happens on a background
        thread so it never stalls generation.
        """
        try:
            cache_b64 = base64.b64encode(cache_to_bytes(past)).decode()
        except Exception:  # noqa: BLE001 - a serialization hiccup must not break generation
            logger.exception("kv: failed to serialize cache; skipping broadcast")
            return
        state = KvState(
            kv_seq=self._kv_seq,
            next_token=next_token,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            cache_b64=cache_b64,
        )
        threading.Thread(
            target=self._send_kv, args=(encode_kv_state(state),), daemon=True
        ).start()

    def _send_kv(self, payload: bytes) -> None:
        for name in self.engine.view.members:
            if name == self.name:
                continue
            with contextlib.suppress(RequestException):
                requests.post(f"{address(name)}/kv", data=payload, timeout=5)

    def store_kv(self, body: bytes) -> None:
        """Store a warm KV-cache snapshot from the leader (freshest wins)."""
        state = decode_kv_state(body)
        if self._kv is None or state.kv_seq >= self._kv.kv_seq:
            self._kv = state

    def _bump_kv_seq(self, kv_seq: int) -> None:
        """Record our global token count in the view so gossip carries it for elections."""
        self.engine.mutate(
            lambda v: ClusterView(
                members={
                    **v.members,
                    self.name: v.members[self.name].model_copy(update={"kv_seq": kv_seq}),
                },
                routing=v.routing,
            )
        )

    # -- gossip transport + failure reaction -------------------------------

    def _send_gossip(self, dest: str | None, msg: dict) -> None:
        """Send an encoded gossip message to ``dest`` (``None`` = broadcast to all peers)."""
        targets = (
            [dest]
            if dest is not None
            else [n for n in self.engine.view.members if n != self.name]
        )
        for target in targets:
            with contextlib.suppress(RequestException):
                requests.post(f"{address(target)}/gossip", json=msg, timeout=2)

    def handle_gossip(self, msg: dict) -> None:
        """Handle an inbound gossip message and send any replies it produces."""
        replies = self.engine.handle(decode_gossip(msg), now=time.monotonic())
        for dest, reply in replies:
            self._send_gossip(dest, encode_gossip(reply))

    def _on_member_dead(self, node_id: str) -> None:
        """A peer was declared dead by the gossip loop: elect a successor or rebalance.

        The engine has already marked ``node_id`` DEAD, so ``should_become_leader`` is
        true (for exactly one node, thanks to the total election order) only when the
        leader died; otherwise, if an expert died and I am leader, rebalance its experts.
        """
        view = self.engine.view
        if should_become_leader(view, self.name):
            logger.info("%s: I win the election; promoting to leader", self.name)
            self.role = NodeRole.LEADER
            self._become_leader()
        elif node_id != self.name and self.role is NodeRole.LEADER:
            self._on_expert_failure(node_id)
