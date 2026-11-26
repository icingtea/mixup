"""Tests for expert placement: rebalance_experts and the least-loaded policy."""

from mixup_core.runtime import ClusterView, LoadMetrics, MemberState, RoutingEntry, RoutingTable
from mixup_core.scheduler import (
    LeastLoadedPolicy,
    balance_experts,
    handle_node_failure,
    rebalance_experts,
)
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    status: MemberStatus = MemberStatus.ALIVE,
    active: int = 0,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        status=status,
        epoch=0,
        incarnation=1,
        priority=1,
        loads=LoadMetrics(active_requests=active),
    )


def _view(*members: MemberState) -> ClusterView:
    return ClusterView(members={m.node_id: m for m in members})


def _table(assignment: dict) -> RoutingTable:
    return RoutingTable(epoch=1, assignment=assignment)


def test_failover_replaces_dead_owner() -> None:
    prev = _table(
        {
            0: RoutingEntry(node_id="n1", slot=0),
            1: RoutingEntry(node_id="n1", slot=1),
            2: RoutingEntry(node_id="n2", slot=0),
        }
    )
    view = _view(_member("n1", status=MemberStatus.DEAD), _member("n2"))
    table = rebalance_experts(view, prev, LeastLoadedPolicy())

    assert set(table.assignment) == {0, 1, 2}  # coverage + uniqueness
    assert table.assignment[0].node_id == "n2"
    assert table.assignment[1].node_id == "n2"
    assert table.assignment[2].node_id == "n2"
    assert table.epoch == prev.epoch + 1


def test_rebalance_keeps_live_owners() -> None:
    prev = _table(
        {
            0: RoutingEntry(node_id="n1", slot=0),
            1: RoutingEntry(node_id="n1", slot=1),
            2: RoutingEntry(node_id="n2", slot=0),
        }
    )
    view = _view(_member("n1"), _member("n2"))
    table = rebalance_experts(view, prev, LeastLoadedPolicy())

    assert table.assignment[0].node_id == "n1"  # unchanged (owner alive)
    assert table.assignment[2].node_id == "n2"
    assert table.epoch == prev.epoch + 1


def test_least_loaded_prefers_under_loaded_node() -> None:
    view = _view(_member("n1", active=5), _member("n2", active=0))
    placed = LeastLoadedPolicy().place(view, [0, 1, 2])
    # n2 stays least-loaded even after 3 placements, so it takes everything.
    assert set(placed.values()) == {"n2"}


def test_least_loaded_balances_when_equal() -> None:
    view = _view(_member("n1", active=0), _member("n2", active=0))
    placed = LeastLoadedPolicy().place(view, [0, 1, 2])
    assert placed[0] == "n1"  # tie-break by node_id
    assert placed[1] == "n2"  # n1 now has one pending
    assert set(placed.values()) == {"n1", "n2"}


def test_slots_recomputed_contiguously() -> None:
    prev = _table(
        {
            0: RoutingEntry(node_id="n1", slot=0),
            5: RoutingEntry(node_id="n1", slot=1),
        }
    )
    view = _view(_member("n1", status=MemberStatus.DEAD), _member("n2"))
    table = rebalance_experts(view, prev, LeastLoadedPolicy())

    assert table.assignment[0].slot == 0
    assert table.assignment[5].slot == 1


def test_handle_node_failure_rebalances() -> None:
    view = ClusterView(
        members={
            "leader": MemberState(
                node_id="leader", role=NodeRole.LEADER, epoch=0, incarnation=1, priority=1
            ),
            "n1": _member("n1"),
            "n2": _member("n2"),
        },
        routing=_table(
            {
                0: RoutingEntry(node_id="n1", slot=0),
                1: RoutingEntry(node_id="n1", slot=1),
                2: RoutingEntry(node_id="n2", slot=0),
            }
        ),
    )

    out = handle_node_failure(view, "n1", "leader")

    assert out.members["n1"].status is MemberStatus.DEAD
    assert out.routing is not None
    assert set(out.routing.assignment) == {0, 1, 2}  # coverage preserved
    assert out.routing.assignment[0].node_id == "n2"  # orphaned experts moved
    assert out.routing.assignment[1].node_id == "n2"
    assert out.routing.assignment[2].node_id == "n2"
    assert out.routing.epoch == view.routing.epoch + 1


def test_balance_experts_gives_rejoined_node_a_share() -> None:
    prev = _table(
        {
            0: RoutingEntry(node_id="n1", slot=0),
            1: RoutingEntry(node_id="n1", slot=1),
            2: RoutingEntry(node_id="n2", slot=0),
            3: RoutingEntry(node_id="n2", slot=1),
        }
    )
    # n3 just rejoined; all three alive experts should now own something.
    view = _view(_member("n1"), _member("n2"), _member("n3"))
    table = balance_experts(view, prev, LeastLoadedPolicy())

    assert set(table.assignment) == {0, 1, 2, 3}
    assert {table.assignment[e].node_id for e in table.assignment} == {"n1", "n2", "n3"}
    assert table.epoch == prev.epoch + 1
