"""Expert placement: the scheduler's pure core.

``rebalance_experts`` recomputes the expert assignment given the current membership and a
``PlacementPolicy``. It is a *deterministic function* — same view → same table — but views
are only eventually consistent, so nodes converge via the table's epoch, not by
commutativity (see docs/plan/phase-6-dynamic-dispatch.md).
"""

from typing import Protocol

from .gossip import Dead, apply_gossip
from .runtime import ClusterView, RoutingEntry, RoutingTable
from .types import ExpertId, MemberStatus, NodeId, NodeRole


class PlacementPolicy(Protocol):
    """Decides which node takes each unowned expert.

    Returns a partial assignment ``expert -> node``; slots are derived later by
    ``rebalance_experts`` from each node's sorted owned list.
    """

    def place(self, view: ClusterView, experts: list[ExpertId]) -> dict[ExpertId, NodeId]:
        ...


class LeastLoadedPolicy:
    """Re-place experts onto the least-loaded alive expert node (the v1 failover default)."""

    def place(self, view: ClusterView, experts: list[ExpertId]) -> dict[ExpertId, NodeId]:
        alive = [
            m
            for m in view.members.values()
            if m.status == MemberStatus.ALIVE and m.role == NodeRole.EXPERT
        ]
        if not experts or not alive:
            return {}

        pending: dict[NodeId, int] = {m.node_id: 0 for m in alive}

        def load(m) -> int:
            return m.loads.active_requests + m.loads.queue_depth + pending[m.node_id]

        result: dict[ExpertId, NodeId] = {}
        for expert in sorted(experts):
            node = min(alive, key=lambda m: (load(m), m.node_id)).node_id
            result[expert] = node
            pending[node] += 1
        return result


def _alive_expert_ids(view: ClusterView) -> set[NodeId]:
    return {
        m.node_id
        for m in view.members.values()
        if m.status == MemberStatus.ALIVE and m.role == NodeRole.EXPERT
    }


def _finalize(placed: dict[ExpertId, NodeId], epoch: int) -> RoutingTable:
    """Derive each node's local slots from its sorted owned list."""
    by_node: dict[NodeId, list[ExpertId]] = {}
    for expert, node in placed.items():
        by_node.setdefault(node, []).append(expert)

    assignment: dict[ExpertId, RoutingEntry] = {}
    for node, experts in by_node.items():
        for slot, expert in enumerate(sorted(experts)):
            assignment[expert] = RoutingEntry(node_id=node, slot=slot)
    return RoutingTable(epoch=epoch, assignment=assignment)


def rebalance_experts(
    view: ClusterView,
    prev: RoutingTable,
    policy: PlacementPolicy,
) -> RoutingTable:
    """Recompute the assignment, re-placing experts whose owner is no longer an alive expert.

    Keeps experts whose current owner is still alive (minimal churn), hands the rest to
    ``policy``, then derives each node's local slots from its sorted owned list and bumps
    the epoch.
    """
    alive = _alive_expert_ids(view)

    kept: dict[ExpertId, NodeId] = {}
    for expert, entry in prev.assignment.items():
        if entry.node_id in alive:
            kept[expert] = entry.node_id

    unowned = [expert for expert in prev.assignment if expert not in kept]
    placed = policy.place(view, unowned)

    return _finalize({**kept, **placed}, prev.epoch + 1)


def balance_experts(
    view: ClusterView,
    prev: RoutingTable,
    policy: PlacementPolicy | None = None,
) -> RoutingTable:
    """Redistribute *all* experts evenly across the alive expert nodes (bump epoch).

    Unlike ``rebalance_experts`` (which only moves orphans), this re-places every expert,
    so it is used when a node *rejoins* — the rejoined node gets a fair share and the
    others shed some of theirs.
    """
    placed = (policy or LeastLoadedPolicy()).place(view, sorted(prev.assignment))
    return _finalize(placed, prev.epoch + 1)


def handle_node_failure(
    view: ClusterView,
    dead_node: NodeId,
    by_node: NodeId,
    policy: PlacementPolicy | None = None,
) -> ClusterView:
    """Mark ``dead_node`` DEAD and rebalance its experts onto the survivors.

    Returns a new view whose routing table (epoch bumped) covers every expert. This is
    the pure core of failover: the central calls it on a connection failure, then
    broadcasts the resulting table to the surviving experts (Phase 6/7).
    """
    member = view.members.get(dead_node)
    incarnation = member.incarnation if member else 1
    view = apply_gossip(
        view,
        Dead(node_id=dead_node, incarnation=incarnation, by_node=by_node),
    )
    if view.routing is not None:
        new_table = rebalance_experts(view, view.routing, policy or LeastLoadedPolicy())
        view = ClusterView(members=view.members, routing=new_table)
    return view
