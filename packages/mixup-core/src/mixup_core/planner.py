"""The placement planner: turn perception (profiles + stats + network) into a routing table.

Thin, pure glue between the placement engine (``placement.py``) and the runtime's
``RoutingTable``. The leader calls ``plan_routing`` at boot and again on the coarse
timescale (membership change / sustained drift) to re-place experts resource-awarely; the
resulting table (epoch bumped) is gossiped/broadcast so nodes reload their share.
"""

from .placement import FrequencyPolicy, LatencyPolicy, PlacementProblem, UniformPolicy
from .runtime import RoutingEntry, RoutingTable
from .stats import ActivationStats
from .types import ExpertId, NodeId

DEFAULT_POLICY = "latency"


def _assignment_from_table(table: RoutingTable | None) -> dict[ExpertId, NodeId] | None:
    if table is None:
        return None
    return {e: entry.node_id for e, entry in table.assignment.items()}


def _finalize(assignment: dict[ExpertId, NodeId], epoch: int) -> RoutingTable:
    """Derive each node's local slots from its sorted owned list."""
    by_node: dict[NodeId, list[ExpertId]] = {}
    for expert, node in assignment.items():
        by_node.setdefault(node, []).append(expert)
    entries: dict[ExpertId, RoutingEntry] = {}
    for node, experts in by_node.items():
        for slot, expert in enumerate(sorted(experts)):
            entries[expert] = RoutingEntry(node_id=node, slot=slot)
    return RoutingTable(epoch=epoch, assignment=entries)


def plan_routing(
    profiles: dict[NodeId, object],
    stats: ActivationStats,
    rtt_ms: dict[NodeId, float],
    leader: NodeId,
    *,
    prev_table: RoutingTable | None = None,
    budget_bytes: dict[NodeId, int] | None = None,
    expert_bytes: int = 75_500_000,
    dense_ms: float = 0.0,
    policy: str = DEFAULT_POLICY,
    num_samples: int = 32,
    max_iters: int = 20,
    seed: int = 0,
    churn_amortize_tokens: int = 1000,
    heterogeneity: dict[NodeId, float] | None = None,
) -> RoutingTable:
    """Compute a resource-aware ``expert -> node`` assignment and wrap it in a table.

    The table's epoch is ``prev_table.epoch + 1`` (or 1 when there is no previous table).
    ``policy`` is ``"latency"`` (default), ``"frequency"``, or ``"uniform"``. ``profiles``
    must include the leader (a candidate owner) and every expert-capable node.
    """
    problem = PlacementProblem(
        num_experts=stats.num_experts,
        profiles=profiles,
        stats=stats,
        rtt_ms=rtt_ms,
        leader=leader,
        expert_bytes=expert_bytes,
        dense_ms=dense_ms,
        budget_bytes=budget_bytes,
        prev=_assignment_from_table(prev_table),
        churn_amortize_tokens=churn_amortize_tokens,
        heterogeneity=heterogeneity or {},
    )
    policy_obj = {
        "latency": LatencyPolicy(max_iters=max_iters, num_samples=num_samples, seed=seed),
        "frequency": FrequencyPolicy(),
        "uniform": UniformPolicy(),
    }[policy]
    assignment = policy_obj.place(problem)
    epoch = (prev_table.epoch + 1) if prev_table is not None else 1
    return _finalize(assignment, epoch)
