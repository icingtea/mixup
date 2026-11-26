"""Expert placement: the scheduler's pure core.

``rebalance_experts`` recomputes the expert assignment given the current membership and a
``PlacementPolicy``. It is a *deterministic function* — same view → same table — but views
are only eventually consistent, so nodes converge via the table's epoch, not by
commutativity (see docs/plan/phase-6-dynamic-dispatch.md).
"""

from typing import Protocol

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
            if m.status is MemberStatus.ALIVE and m.role is NodeRole.EXPERT
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
        if m.status is MemberStatus.ALIVE and m.role is NodeRole.EXPERT
    }


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

    final: dict[ExpertId, NodeId] = {**kept, **placed}

    by_node: dict[NodeId, list[ExpertId]] = {}
    for expert, node in final.items():
        by_node.setdefault(node, []).append(expert)

    assignment: dict[ExpertId, RoutingEntry] = {}
    for node, experts in by_node.items():
        for slot, expert in enumerate(sorted(experts)):
            assignment[expert] = RoutingEntry(node_id=node, slot=slot)

    return RoutingTable(epoch=prev.epoch + 1, assignment=assignment)
