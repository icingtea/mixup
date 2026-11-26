"""The node runtime: one process that can be a leader (generate) or an expert (serve).

Holds the role-dependent state (expert group, dense model, dispatcher) and runs the
control loop: leaders generate, followers run a watchdog that promotes them on leader
failure (Phases 3/4). The FastAPI app (built in infer.py) reads this state.
"""

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
from mixup_core.gossip import Dead, apply_gossip
from mixup_core.kvledger import Checkpoint, decode_checkpoint, encode_checkpoint
from mixup_core.runtime import ClusterView, RoutingTable
from mixup_core.scheduler import LeastLoadedPolicy, handle_node_failure, rebalance_experts
from mixup_core.types import MemberStatus, NodeRole

from .dispatch import Dispatcher
from .model import ExpertGroup, load_central

logger = logging.getLogger(__name__)


class _CheckpointProcessor(LogitsProcessor):
    """Writes the token-sequence checkpoint after each generated token (before sampling)."""

    def __init__(self, prompt_len: int, path) -> None:
        self.prompt_len = prompt_len
        self.path = path

    def __call__(self, input_ids, scores):
        kv_seq = input_ids.shape[1] - self.prompt_len
        checkpoint = Checkpoint(kv_seq=kv_seq, input_ids=input_ids[0].tolist())
        self.path.write_text(encode_checkpoint(checkpoint).decode())
        return scores


def broadcast_routing(view: ClusterView, by_node: str) -> None:
    """Push the current routing table to every alive expert node (so they load new experts)."""
    table = view.routing
    if table is None:
        return
    payload = table.model_dump()
    for member in view.members.values():
        if member.role is NodeRole.EXPERT and member.status is MemberStatus.ALIVE:
            try:
                requests.post(f"{address(member.node_id)}/routing", json=payload, timeout=60)
                logger.info("broadcast: sent routing epoch %d to %s", table.epoch, member.node_id)
            except RequestException:
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
        """Entry point for the background control thread."""
        if self.role is NodeRole.LEADER:
            self._become_leader()
            self._generate()
        else:
            self._load_experts()
            self._watchdog()

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

    def _generate(self, resume: bool = False) -> None:
        assert self.model is not None and self.tokenizer is not None
        assert self.dispatcher is not None
        self.dispatcher.wait_for_experts()

        if resume and CHECKPOINT_FILE.exists():
            checkpoint = decode_checkpoint(CHECKPOINT_FILE.read_bytes())
            ids = torch.tensor([checkpoint.input_ids])
            prompt_len = len(checkpoint.input_ids) - checkpoint.kv_seq
            remaining = max(0, self.env.max_new_tokens - checkpoint.kv_seq)
            logger.info(
                "resuming from checkpoint: kv_seq=%d, %d tokens left",
                checkpoint.kv_seq,
                remaining,
            )
        else:
            ids = self.tokenizer(self.env.prompt, return_tensors="pt").input_ids
            prompt_len = ids.shape[1]
            remaining = self.env.max_new_tokens

        if remaining <= 0:
            logger.info("generation already complete")
            return

        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        processor = _CheckpointProcessor(prompt_len, CHECKPOINT_FILE)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=remaining, do_sample=False, logits_processor=[processor]
            )
        logger.info("generated: %s", self.tokenizer.decode(out[0][prompt_len:]))

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
                self._generate(resume=True)
                return
