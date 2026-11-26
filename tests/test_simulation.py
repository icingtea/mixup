"""Deterministic simulation: drive gossip + election + scheduler together.

These tests compose the pure state machines (apply_gossip, elect, rebalance_experts) to
verify the cluster invariants from the plan (coverage, uniqueness, at-most-one-leader),
with no network — a miniature version of the Phase-9 chaos harness, but fully in-core.
"""

from mixup_core.election import elect
from mixup_core.gossip import Alive, Dead, Suspect, apply_gossip
from mixup_core.runtime import ClusterView, MemberState, RoutingEntry, RoutingTable
from mixup_core.scheduler import LeastLoadedPolicy, rebalance_experts
from mixup_core.types import MemberStatus, NodeRole

NUM_EXPERTS = 24


def _member(
    node: str,
    role: NodeRole = NodeRole.EXPERT,
    status: MemberStatus = MemberStatus.ALIVE,
    priority: int = 1,
    kv_seq: int = 0,
    incarnation: int = 1,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=role,
        status=status,
        epoch=0,
        incarnation=incarnation,
        priority=priority,
        kv_seq=kv_seq,
    )


def _view(*members: MemberState) -> ClusterView:
    return ClusterView(members={m.node_id: m for m in members})


def _table(assignment: dict) -> RoutingTable:
    return RoutingTable(epoch=1, assignment=assignment)


def _contiguous_table(blocks: dict[str, tuple[int, int]]) -> RoutingTable:
    """Build a table where each node owns a contiguous [lo, hi] range of experts."""
    assignment: dict[int, RoutingEntry] = {}
    for node, (lo, hi) in blocks.items():
        for slot, expert in enumerate(range(lo, hi + 1)):
            assignment[expert] = RoutingEntry(node_id=node, slot=slot)
    return _table(assignment)


def test_expert_crash_rebalances_with_coverage_and_uniqueness() -> None:
    view = _view(
        _member("leader", role=NodeRole.LEADER),
        _member("n1"),
        _member("n2"),
        _member("n3"),
    )
    prev = _contiguous_table({"n1": (0, 7), "n2": (8, 15), "n3": (16, 23)})

    # n1 dies.
    view = apply_gossip(view, Dead(node_id="n1", incarnation=1, by_node="n2"))

    table = rebalance_experts(view, prev, LeastLoadedPolicy())

    owners = [e.node_id for e in table.assignment.values()]
    assert set(table.assignment) == set(range(NUM_EXPERTS))  # coverage
    assert "n1" not in owners  # dead node owns nothing
    assert "leader" not in owners  # leader holds no experts
    assert set(owners) <= {"n2", "n3"}  # only alive expert nodes own experts


def test_leader_crash_elects_freshest_survivor() -> None:
    view = _view(
        _member("leader", role=NodeRole.LEADER, kv_seq=10),
        _member("n1", kv_seq=8),
        _member("n2", kv_seq=9),
    )
    view = apply_gossip(view, Dead(node_id="leader", incarnation=1, by_node="n1"))
    assert elect(view) == "n2"  # highest kv_seq among the alive


def test_partition_stale_message_is_ignored() -> None:
    view = _view(
        _member("leader", role=NodeRole.LEADER, incarnation=5, kv_seq=10),
        _member("n1", kv_seq=3),
    )
    # A stale Dead (older incarnation) about a resurrected leader must not kill it.
    view = apply_gossip(view, Dead(node_id="leader", incarnation=4, by_node="n1"))
    assert view.members["leader"].status is MemberStatus.ALIVE
    assert elect(view) == "leader"


def test_resurrection_after_rebalance() -> None:
    view = _view(
        _member("leader", role=NodeRole.LEADER),
        _member("n1"),
        _member("n2"),
    )
    prev = _contiguous_table({"n1": (0, 11), "n2": (12, 23)})

    view = apply_gossip(view, Suspect(node_id="n1", incarnation=1, by_node="n2"))
    assert view.members["n1"].status is MemberStatus.SUSPECT

    # n1 recovers with a fresh incarnation (a restart), clearing the suspicion.
    view = apply_gossip(view, Alive(node_id="n1", incarnation=2, epoch=1))
    assert view.members["n1"].status is MemberStatus.ALIVE

    # A rebalance while n1 is alive keeps its experts (no churn).
    table = rebalance_experts(view, prev, LeastLoadedPolicy())
    assert table.assignment[0].node_id == "n1"
    assert table.assignment[23].node_id == "n2"
