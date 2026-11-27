"""The node runtime: one process that can be a leader (generate) or an expert (serve).

Holds the role-dependent state (expert group, dense model, dispatcher) and runs the
control loop: leaders generate, followers run a watchdog that promotes them on leader
failure (Phases 3/4). The FastAPI app (built in infer.py) reads this state.
"""

import base64
import contextlib
import logging
import threading
import time

import requests
import torch
from requests.exceptions import RequestException
from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM

from mixup_core.constants import MODEL_DIR, MODEL_FILE, address
from mixup_core.devices import DeviceProfile, ProfileMode
from mixup_core.election import (
    apply_leader_announcement,
    current_leader,
    promote,
    should_become_leader,
)
from mixup_core.env import NodeEnv
from mixup_core.gossip import Ack, Alive, apply_gossip, decode_gossip, encode_gossip
from mixup_core.gossip_engine import GossipEngine
from mixup_core.kvledger import KvState, decode_kv_state, encode_kv_state
from mixup_core.planner import plan_routing
from mixup_core.predictor import PredictorState
from mixup_core.replica import failover_with_backups, plan_backups
from mixup_core.resource import heterogeneity
from mixup_core.runtime import ClusterView, RoutingTable
from mixup_core.scheduler import (
    LeastLoadedPolicy,
    balance_experts,
)
from mixup_core.stats import TraceBuffer
from mixup_core.tokenlog import CommitLog, CommitRecord, CommitTracker
from mixup_core.types import MemberStatus, NodeRole

from .dispatch import Dispatcher
from .gossip_loop import GossipLoop
from .kv_cache import cache_from_bytes, cache_to_bytes
from .memory import TieredExpertStore
from .metrics import MetricsCollector
from .model import ExpertGroup, load_central
from .monitor import ResourceMonitor
from .predictor import LRPredictor, MLPPredictor

logger = logging.getLogger(__name__)

# SWIM gossip timing. Ping/suspect timeouts are long enough to tolerate a peer reloading
# experts (a GIL-heavy torch load of several seconds) without a spurious DEAD, which would
# otherwise trigger an unnecessary failover.
GOSSIP_INTERVAL = 0.5
PING_TIMEOUT = 5.0
SUSPECT_TIMEOUT = 5.0
INDIRECT_K = 2

DEAD_REPROBE_INTERVAL = 5.0
"""How often (seconds) to re-probe a dead peer when no alive peer remains. A transient
stall (e.g. an expert reload holding the GIL) can make a node mark a still-alive peer
DEAD; re-probing it lets the peer's ack resurrect it instead of leaving a permanent
partition. This is what heals the mutual-death / "no peer to ping" livelock."""

REPLAN_MIN_TOKENS = 50
"""Minimum observed tokens before the leader re-places experts from activation stats."""

REPLAN_INTERVAL_TOKENS = 200
"""Re-evaluate placement every this many observed tokens (after the first re-plan)."""


def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Sample the next token via top-k sampling (greedy when temperature <= 0 or top_k <= 1).

    Returns a 1-D ``[batch]`` tensor, matching ``argmax``'s shape.
    """
    if temperature <= 0.0 or top_k <= 1:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    k = min(top_k, scaled.shape[-1])
    topk_vals, topk_idx = torch.topk(scaled, k, dim=-1)
    probs = torch.softmax(topk_vals, dim=-1)
    sampled = torch.multinomial(probs, 1)  # [batch, 1]
    return topk_idx.gather(-1, sampled).squeeze(-1)  # [batch]


def _assignment_summary(table: RoutingTable) -> dict[str, int]:
    """Per-node expert counts, for logging placement changes."""
    counts: dict[str, int] = {}
    for entry in table.assignment.values():
        counts[entry.node_id] = counts.get(entry.node_id, 0) + 1
    return dict(sorted(counts.items()))


def broadcast_routing(view: ClusterView, by_node: str) -> None:
    """Push the current routing table to every expert node (so they load new experts).

    Iterates the routing table's expert nodes (not the membership roster), so a node that
    was rebalanced away or died is not contacted. Fire-and-forget on a background thread so
    a slow or dead peer never stalls the caller (the gossip loop or the /join handler); the
    nodes reload asynchronously via /routing anyway.
    """
    table = view.routing
    if table is None:
        return
    payload = table.model_dump()
    nodes = sorted({entry.node_id for entry in table.assignment.values()})
    threading.Thread(
        target=_broadcast_routing, args=(payload, nodes, table.epoch), daemon=True
    ).start()


def _broadcast_routing(payload: dict, nodes: list[str], epoch: int) -> None:
    """Best-effort push of ``payload`` to ``nodes`` (2 short retries each)."""
    for node_id in nodes:
        sent = False
        for _ in range(2):
            try:
                requests.post(f"{address(node_id)}/routing", json=payload, timeout=2)
                logger.info("broadcast: sent routing epoch %d to %s", epoch, node_id)
                sent = True
                break
            except RequestException:
                time.sleep(0.5)
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
            choose_peer=self._choose_peer,
            on_dead=self._on_member_dead,
        )
        # The freshest warm KV-cache snapshot this node has received (as a follower), for
        # resuming an interrupted generation after a failover.
        self._kv: KvState | None = None
        self._kv_seq = 0  # global monotonic token count (election freshness + dedup)
        self.kv_sync_every = env.config.kv_sync_every
        self._peer_idx = 0  # round-robin cursor for gossip peer selection
        self._dead_reprobe_at = 0.0  # next time we may re-probe a dead peer
        self._no_peer_statuses: dict[str, str] | None = None  # last logged "no peer" state
        self._trace: TraceBuffer | None = None  # leader's activation stats (Phase 2)
        self._commit_log = CommitLog()  # replicated commit log (Phase 5)
        self._commit: CommitTracker | None = None  # leader's commit quorum (Phase 5)
        self._next_replan_at = REPLAN_MIN_TOKENS  # token count for the next re-plan
        self.monitor = ResourceMonitor()  # live CPU/memory/GPU sampling
        self.predictor = None  # leader's next-layer predictor (Phase 6)
        self._predictor_version = 0  # version for predictor weight sync (Phase 6)
        self._predictor_state: PredictorState | None = None  # last synced (follower)

    # -- control loop ------------------------------------------------------

    def run(self) -> None:
        """Entry point for the background control thread.

        Boots the node, then starts the SWIM gossip loop. The loop starts *after* model
        loading so its probes don't race the GIL-heavy torch load (which would cause
        spurious SUSPECTs at boot). A restarted node's revival doesn't depend on its own
        loop timing: its fresh (higher) incarnation revives it on the peers' side.
        """
        self._discover_topology()
        if self.role == NodeRole.LEADER:
            self._become_leader()
        else:
            self._load_experts()
            self._declare_rejoin()
        threading.Thread(target=self.loop.run, args=(0.1,), daemon=True).start()

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
        """Load the dense model and (if we own experts) keep/load our experts, then lead.

        Leader-as-owner (Phase 1): instead of shedding our experts on promotion, we keep
        them and serve them alongside the dense model. We promote in the view *first*
        (bump epoch + set role), announce immediately, and only then load the dense model;
        ``ready`` flips at the end, so a client polling ``/health`` waits for the load
        instead of hammering ``/generate`` with 503s.
        """
        view = self.engine.mutate(lambda v: promote(v, self.name))
        assert view.routing is not None
        self.role = NodeRole.LEADER
        self._commit = CommitTracker(len(view.members), log=self._commit_log)
        if self.expert_group is None:
            self.expert_group = self._make_expert_group(view.routing)
        self._trace = TraceBuffer(
            self.model_config.num_hidden_layers, self.model_config.num_local_experts or 0
        )
        # Phase 6: the leader trains a next-layer predictor online and syncs its weights.
        num_experts = self.model_config.num_local_experts or 0
        feature_dim = self.model_config.hidden_size + num_experts + 1
        self.predictor = self._make_predictor(feature_dim, num_experts)
        self._predictor_version = 0
        self.dispatcher = Dispatcher(
            self.name,
            view.routing,
            self._on_expert_failure,
            stats=self._trace,
            predictor=self.predictor,
            num_experts=num_experts,
        )
        self._announce_leader()
        self.model = load_central(self.model_config, MODEL_FILE, self.dispatcher)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.ready = True
        logger.info(
            "%s: dense model loaded; I am leader (holding %d experts)",
            self.name,
            len(self.expert_group.experts),
        )

    def _load_experts(self) -> None:
        view = self.engine.view
        assert view.routing is not None
        self.expert_group = self._make_expert_group(view.routing)
        self.ready = True
        logger.info("%s: %d experts ready", self.name, len(self.expert_group.experts))

    def _make_expert_group(self, table) -> ExpertGroup | TieredExpertStore:
        """Pick the expert store: tiered (with residency) when a budget is set, else resident."""
        profile = self.env.profiles.get(self.name)
        budget = profile.expert_budget_bytes if profile is not None else None
        if budget is not None:
            store = TieredExpertStore(self.name, self.model_config, MODEL_FILE, profile, budget)
            store.set_routing(table)
            return store
        return ExpertGroup(self.name, self.model_config, MODEL_FILE, table)

    def _make_predictor(self, feature_dim: int, num_experts: int):
        """The leader's next-layer predictor (configurable: ``predictor_kind``).

        CoMoE specifies a 2-layer MLP; the logistic-regression baseline is kept as an
        option (it won the eval, but the paper's default is the MLP).
        """
        kind = getattr(self.env.config, "predictor_kind", "mlp")
        if kind == "mlp":
            return MLPPredictor(feature_dim, num_experts)
        return LRPredictor(feature_dim, num_experts)

    def _sync_predictor(self) -> None:
        """Broadcast the predictor weights to followers so a promoted leader resumes them."""
        if self.predictor is None:
            return
        self._predictor_version += 1
        payload = self.predictor.to_state(self._predictor_version).model_dump()
        for name in self.engine.view.members:
            if name == self.name:
                continue
            threading.Thread(
                target=self._post_predictor, args=(name, payload), daemon=True
            ).start()

    def _post_predictor(self, name: str, payload: dict) -> None:
        with contextlib.suppress(RequestException):
            requests.post(f"{address(name)}/predictor", json=payload, timeout=5)

    def store_predictor(self, body: dict) -> None:
        """Store synced predictor weights (follower); freshest version wins."""
        state = PredictorState.model_validate(body)
        if self._predictor_state is not None and state.version <= self._predictor_state.version:
            return
        self._predictor_state = state
        if state.kind == "lr":
            self.predictor = LRPredictor.from_state(state)
        elif state.kind == "mlp":
            self.predictor = MLPPredictor.from_state(state)

    def predictor_info(self) -> dict:
        """Live predictor state + accuracy for the observability UI."""
        if self.predictor is None:
            return {"kind": "none", "version": 0, "samples": 0, "top1_hits": 0, "top3_hits": 0}
        return {
            "kind": getattr(self.predictor, "kind", "?"),
            "version": self._predictor_version,
            "samples": getattr(self.predictor, "samples", 0),
            "top1_hits": getattr(self.predictor, "top1_hits", 0),
            "top3_hits": getattr(self.predictor, "top3_hits", 0),
            "feature_dim": getattr(self.predictor, "feature_dim", 0),
            "num_experts": getattr(self.predictor, "num_experts", 0),
            "hidden_dim": getattr(self.predictor, "hidden_dim", 0),
            "learning_rate": getattr(self.predictor, "lr", 0.0),
        }

    def expert_stats(self) -> dict:
        """Per-expert activation counts (summed over layers) for the observability UI."""
        if self._trace is None:
            return {"counts": [], "num_experts": 0}
        stats = self._trace.snapshot()
        counts = [
            sum(stats.counts[ly][e] for ly in range(stats.num_layers))
            for e in range(stats.num_experts)
        ]
        return {"counts": counts, "num_experts": stats.num_experts}

    def _on_expert_failure(self, node: str) -> RoutingTable:
        def _fail(v: ClusterView) -> ClusterView:
            v = failover_with_backups(v, node, by_node=self.name)
            # Update the dispatcher's table atomically with the view (under the lock), so
            # it can never diverge from the routing the view just adopted.
            if self.dispatcher is not None:
                self.dispatcher.set_table(v.routing)
            return v

        view = self.engine.mutate(_fail)
        new_table = view.routing
        if new_table is not None:
            if self.expert_group is not None:
                self.expert_group.set_routing(new_table)
            broadcast_routing(view, self.name)
        assert new_table is not None
        return new_table

    def adopt_routing(self, table: RoutingTable) -> bool:
        """Adopt a routing table and propagate it to the dispatcher + expert group.

        The view and the dispatcher's table are updated *atomically* under the engine lock
        (so the dispatcher can never diverge from the view), then the expert group is
        reloaded and the table broadcast. Returns ``True`` if the table was adopted (it was
        newer than the view's current routing).
        """

        def _on_adopted(t: RoutingTable) -> None:
            if self.dispatcher is not None:
                self.dispatcher.set_table(t)

        if not self.engine.adopt_routing(table, on_adopted=_on_adopted):
            return False
        if self.expert_group is not None:
            self.expert_group.set_routing(table)
        broadcast_routing(self.engine.view, self.name)
        return True

    def _effective_profiles(self) -> dict[str, DeviceProfile]:
        """Profiles with online cost correction applied (per ``profile_mode``).

        ``configured`` trusts the static profile; ``measured`` trusts the online fit;
        ``hybrid`` blends the two (seeded from the profile, corrected toward the measured
        fit). The fit comes from the dispatcher's measured ``(n_experts, server_ms)``
        samples, which are exactly what ``expert_fixed_ms + n * expert_per_ms`` models.
        """
        if self.env.profile_mode is ProfileMode.CONFIGURED or self.dispatcher is None:
            return self.env.profiles
        measured = self.dispatcher.cost_estimates()
        out: dict[str, DeviceProfile] = {}
        for name, p in self.env.profiles.items():
            fit = measured.get(name)
            if fit is None:
                out[name] = p
                continue
            fixed, per = fit
            if self.env.profile_mode is ProfileMode.MEASURED:
                fixed, per = max(0.0, fixed), max(0.0, per)
            else:  # HYBRID: average of declared and measured
                fixed = (p.expert_fixed_ms + max(0.0, fixed)) / 2.0
                per = (p.expert_per_ms + max(0.0, per)) / 2.0
            out[name] = p.model_copy(update={"expert_fixed_ms": fixed, "expert_per_ms": per})
        return out

    def _heterogeneity(self, profiles: dict[str, DeviceProfile]) -> dict[str, float]:
        """Per-device heterogeneity ``H_d`` from the (corrected) profiles.

        Each device contributes compute, serving latency, bandwidth, memory, and cores;
        ``H_d`` is the mean normalized difference from its peers (0 = identical).
        """
        resources: dict[str, dict[str, float]] = {
            name: {
                "compute": float(p.gpu_tflops or p.cpu_gflops),
                "latency": float(p.expert_per_ms),
                "bandwidth": float(p.disk_read_mbps),
                "memory": float(p.vram_bytes or p.ram_bytes),
                "cores": float(p.cpu_cores),
            }
            for name, p in profiles.items()
        }
        return heterogeneity(resources)

    def replan(self) -> RoutingTable | None:
        """Recompute expert placement from observed activations + profiles (Phase 2).

        Returns the new table (epoch bumped) if it changed, else ``None``. Skipped when the
        cluster has no profiles (classic single-node-GPU setup) or too few observed tokens.
        """
        if not self.env.profiles:
            logger.info("%s: replan skipped (no profiles)", self.name)
            return None
        if self._trace is None:
            logger.info("%s: replan skipped (no trace)", self.name)
            return None
        stats = self._trace.snapshot()
        if stats.total_tokens < REPLAN_MIN_TOKENS:
            logger.info(
                "%s: replan skipped (only %d tokens)", self.name, stats.total_tokens
            )
            return None
        view = self.engine.view
        assert view.routing is not None
        budgets = {
            n: p.expert_budget_bytes
            for n, p in self.env.profiles.items()
            if p.expert_budget_bytes is not None
        } or None
        try:
            profiles = self._effective_profiles()
            rtt_ms = dict(self.dispatcher.rtt_ms) if self.dispatcher else {}
            new_table = plan_routing(
                profiles,
                stats,
                rtt_ms={n: rtt_ms.get(n, 0.0) for n in profiles},
                leader=self.name,
                prev_table=view.routing,
                budget_bytes=budgets,
                heterogeneity=self._heterogeneity(profiles),
            )
        except ValueError as exc:
            logger.warning("%s: replan skipped: %s", self.name, exc)
            return None
        # Assign replica backups (Phase 4): each expert gets a backup on a different node.
        alive = {
            m.node_id for m in self.engine.view.members.values() if m.status == MemberStatus.ALIVE
        }
        new_table = plan_backups(new_table, alive)
        if new_table.assignment == view.routing.assignment:
            logger.info(
                "%s: replan produced no change (%s)",
                self.name,
                _assignment_summary(new_table),
            )
            return None
        # Adopt atomically (view + dispatcher under the engine lock), skipping if a
        # concurrent failover/rejoin has already advanced the epoch.
        if not self.adopt_routing(new_table):
            logger.info("%s: replan result stale, skipped", self.name)
            return None
        logger.info(
            "%s: replanned from %d observed tokens -> routing epoch %d (%s)",
            self.name,
            stats.total_tokens,
            new_table.epoch,
            _assignment_summary(new_table),
        )
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
        me = self.engine.view.members[self.name]
        try:
            requests.post(
                f"{address(leader)}/join",
                json={"node_id": self.name, "incarnation": me.incarnation},
                timeout=5,
            )
            logger.info("%s: declared rejoin to leader %s", self.name, leader)
        except RequestException:
            logger.warning("%s: could not declare rejoin to %s", self.name, leader)

    def rejoin(self, node_id: str, incarnation: int | None = None) -> None:
        """A node announced it rejoined: mark it alive and give it a share of experts.

        With warm stats, re-place via the churn-aware planner (gives the rejoined node a
        fair share without reshuffling everyone); at cold start, fall back to an even
        reshuffle.
        """
        member = self.engine.view.members.get(node_id)
        if member is None:
            return
        # Prefer the rejoining node's own (fresh) incarnation; fall back to bumping ours.
        inc = incarnation if incarnation is not None else member.incarnation + 1

        if self._trace is not None and self._trace.tokens >= REPLAN_MIN_TOKENS:
            self.engine.mutate(
                lambda v: apply_gossip(
                    v,
                    Alive(node_id=node_id, incarnation=inc, epoch=member.epoch),
                )
            )
            if self.replan() is not None:
                logger.info(
                    "%s: rebalanced after %s rejoined (via replan)", self.name, node_id
                )
                return

        def _rebalance(v: ClusterView) -> ClusterView:
            v = apply_gossip(
                v,
                Alive(node_id=node_id, incarnation=inc, epoch=member.epoch),
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
        # Apply the Granite chat template (the model is instruct-tuned; a bare prompt
        # produces degenerate output). Falls back to raw tokenization if templating fails.
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )["input_ids"].to(device)
        except Exception:  # noqa: BLE001 - older tokenizers / missing Jinja helpers
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
            nxt = torch.tensor([state.next_token], device=device)
            full = state.input_ids[prompt_len:]
            # Resume from the full warm snapshot, not the commit-log watermark. The KV
            # snapshot is broadcast every ``kv_sync_every`` tokens and is a self-consistent
            # checkpoint (input_ids + cache + next token); the commit log is only replicated
            # at the *end* of generation (commit-before-emit), so ``committed`` is 0 during
            # a mid-generation failover. Truncating to it dropped every already-generated
            # token and returned a mid-sentence fragment.
            generated_ids: list[int] = list(full)
            remaining = max(0, max_new_tokens - len(generated_ids))
            logger.info(
                "resuming warm cache: %d of %d tokens already generated, %d left",
                len(full),
                max_new_tokens,
                remaining,
            )
            if remaining <= 0:
                # Already fully generated (the response was lost in flight): return the
                # complete text rather than an empty string.
                logger.info("generation already complete")
                return self.tokenizer.decode(full), len(full), 0.0, 0.0, 0.0
        else:
            past = None
            nxt = None
            generated_ids = []
            remaining = max_new_tokens

        metrics = MetricsCollector(self.name, prompt, max_new_tokens)
        self.dispatcher.metrics = metrics

        eos = self.tokenizer.eos_token_id
        temperature = getattr(self.env.config, "temperature", 0.7)
        top_k = getattr(self.env.config, "top_k", 50)
        start = time.perf_counter()
        with torch.no_grad():
            if past is None:
                out = self.model(input_ids=prompt_ids, cache=None, use_cache=True)
                past = out.cache
                nxt = sample_token(out.logits[:, -1, :], temperature, top_k)
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
                    self._sync_predictor()
                generated_ids.append(nxt.item())
                self._kv_seq += 1
                self._bump_kv_seq(self._kv_seq)
                if self._commit is not None:
                    # Append to the commit log and ack our own copy (the leader is one of
                    # the quorum). Followers ack via /commit; the watermark advances on a
                    # majority, and resume truncates to it.
                    epoch = self.engine.view.members[self.name].epoch
                    self._commit.append(epoch, nxt.item(), self._kv_seq)
                    self._commit.record_ack(self._commit.log.next_index - 1, self.name)
                out = self.model(input_ids=nxt.unsqueeze(0), cache=past, use_cache=True)
                past = out.cache
                nxt = sample_token(out.logits[:, -1, :], temperature, top_k)

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

        # Phase 5: replicate the commit log to followers and advance the committed
        # watermark on majority-ack before returning (commit-before-emit).
        committed = self._replicate_commit(
            past, nxt.item(), prompt_list + generated_ids, max_new_tokens
        )
        if committed < generated:
            logger.warning(
                "commit quorum: only %d/%d tokens committed", committed, generated
            )

        # Phase 2/4 coarse loop: re-place experts periodically as activation stats drift.
        # The placement objective amortizes churn, so re-planning only moves experts when
        # the latency gain is worth the reload cost (anti-oscillation).
        if self._trace is not None and self._trace.tokens >= self._next_replan_at:
            self._next_replan_at = self._trace.tokens + REPLAN_INTERVAL_TOKENS
            self.replan()

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
            epoch=self.engine.view.members[self.name].epoch,
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
        """Store a warm KV-cache snapshot from the leader (freshest wins).

        Ordering is ``(epoch, kv_seq)``: a promoted leader's ``kv_seq`` restarts at 0 but
        its epoch is higher, so its snapshots still replace the dead leader's.
        """
        state = decode_kv_state(body)
        if self._kv is None or (state.epoch, state.kv_seq) >= (self._kv.epoch, self._kv.kv_seq):
            self._kv = state
            logger.info(
                "kv: stored warm snapshot (epoch=%d kv_seq=%d)", state.epoch, state.kv_seq
            )

    def store_commit(self, body: dict) -> int:
        """Store replicated commit records + the aligned KV snapshot (follower).

        Returns the highest record index stored (the leader counts this as an ack through
        that index). The KV snapshot is stored via the same freshest-wins rule as
        ``store_kv``, so it lands at the committed watermark.
        """
        records = [CommitRecord(**r) for r in body.get("records", [])]
        last = 0
        for record in records:
            highest = self._commit_log.records[-1].index if self._commit_log.records else 0
            if record.index > highest:
                self._commit_log.append(record)
            last = max(last, record.index)
        if body.get("kv"):
            state = decode_kv_state(body["kv"].encode())
            if self._kv is None or (state.epoch, state.kv_seq) >= (self._kv.epoch, self._kv.kv_seq):
                self._kv = state
        return last

    def _replicate_commit(self, past, next_token, input_ids, max_new_tokens) -> int:
        """Replicate the commit log + the aligned KV snapshot to followers; return the
        committed watermark (majority-ack of the whole log)."""
        if self._commit is None or not self._commit.log.records:
            return 0
        last = self._commit.log.records[-1].index
        cache_b64 = base64.b64encode(cache_to_bytes(past)).decode()
        kv = KvState(
            kv_seq=self._kv_seq,
            next_token=next_token,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            cache_b64=cache_b64,
            epoch=self.engine.view.members[self.name].epoch,
        )
        payload = {
            "records": [r.model_dump() for r in self._commit.log.records],
            "last_index": last,
            "kv": encode_kv_state(kv).decode(),
        }
        for name in self.engine.view.members:
            if name == self.name:
                continue
            try:
                resp = requests.post(f"{address(name)}/commit", json=payload, timeout=5)
                acked = resp.json().get("acked", 0)
                if acked > 0:
                    self._commit.record_acks_through(acked, name)
            except (RequestException, ValueError):
                pass
        return self._commit.log.committed

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

    def handle_gossip(self, msg: dict) -> None:
        """Handle an inbound gossip message and send any replies it produces.

        Replies go out asynchronously (the FastAPI handler runs on the event loop, and a
        synchronous POST could deadlock two nodes pinging each other at once).
        """
        decoded = decode_gossip(msg)
        replies = self.engine.handle(decoded, now=time.monotonic())
        if isinstance(decoded, Ack):
            self.loop.on_ack(decoded.node_id)
        for dest, reply in replies:
            self._send_gossip(dest, encode_gossip(reply))

    def _choose_peer(self, exclude: frozenset[str]) -> str | None:
        """Round-robin over the alive peers so every node gets probed (not just the first).

        When no alive peer remains, re-probe a dead peer at a reduced rate so a falsely-dead
        node can revive (its ack resurrects it). The "no peer" situation is only logged when
        it changes, so an isolated node doesn't flood the log every gossip interval.
        """
        members = self.engine.view.members
        alive = [
            m.node_id
            for m in members.values()
            if m.status == MemberStatus.ALIVE
            and m.node_id != self.name
            and m.node_id not in exclude
        ]
        if alive:
            peer = alive[self._peer_idx % len(alive)]
            self._peer_idx += 1
            return peer

        dead = [
            m.node_id
            for m in members.values()
            if m.status == MemberStatus.DEAD
            and m.node_id != self.name
            and m.node_id not in exclude
        ]
        if dead and time.monotonic() >= self._dead_reprobe_at:
            self._dead_reprobe_at = time.monotonic() + DEAD_REPROBE_INTERVAL
            peer = dead[self._peer_idx % len(dead)]
            self._peer_idx += 1
            return peer

        statuses = {m.node_id: m.status.value for m in members.values()}
        if statuses != self._no_peer_statuses:
            self._no_peer_statuses = statuses
            logger.info(
                "gossip: no alive peer to ping (statuses=%s exclude=%s)",
                statuses,
                sorted(exclude),
            )
        return None

    def _send_gossip(self, dest: str | None, msg: dict) -> None:
        """Send an encoded gossip message to ``dest`` (``None`` = broadcast to all peers).

        Fire-and-forget: the POST runs on a background thread so the gossip loop never
        stalls on a slow or dead peer (a synchronous 2s timeout would delay failure
        detection).
        """
        targets = (
            [dest]
            if dest is not None
            else [n for n in self.engine.view.members if n != self.name]
        )
        threading.Thread(target=self._post_gossip, args=(targets, msg), daemon=True).start()

    def _post_gossip(self, targets: list[str], msg: dict) -> None:
        for target in targets:
            with contextlib.suppress(RequestException):
                requests.post(f"{address(target)}/gossip", json=msg, timeout=2)

    def _on_member_dead(self, node_id: str) -> None:
        """A peer was declared dead by the gossip loop: elect a successor or rebalance.

        The engine has already marked ``node_id`` DEAD, so ``should_become_leader`` is
        true (for exactly one node, thanks to the total election order) only when the
        leader died; otherwise, if an expert died and I am leader, rebalance its experts.
        """
        view = self.engine.view
        if should_become_leader(view, self.name):
            logger.info("%s: I win the election; promoting to leader", self.name)
            self.ready = False  # not ready to generate until the dense model is loaded
            # Re-home the dead leader's experts *before* promoting (and broadcast the new
            # routing so surviving experts adopt it), otherwise the first generation's
            # dispatcher points at the dead node and the survivors skip the re-homed experts.
            self._on_expert_failure(node_id)
            self._become_leader()
        elif node_id != self.name and self.role == NodeRole.LEADER:
            self._on_expert_failure(node_id)
        else:
            logger.info(
                "%s: %s is dead; no action (leader=%s role=%s)",
                self.name,
                node_id,
                current_leader(view),
                self.role.value,
            )
