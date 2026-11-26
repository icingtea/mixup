"""Tests for the live runtime state: membership, routing table, aggregate view."""

from mixup_core.runtime import ClusterView, LoadMetrics, MemberState, RoutingEntry, RoutingTable
from mixup_core.types import NodeRole


def _member(node: str, priority: int = 1, kv_seq: int = 0) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        epoch=0,
        incarnation=1,
        priority=priority,
        kv_seq=kv_seq,
    )


def test_load_metrics_defaults() -> None:
    metrics = LoadMetrics()
    assert metrics.active_requests == 0
    assert metrics.queue_depth == 0
    assert metrics.free_slots == 0


def test_member_defaults() -> None:
    member = _member("n1")
    assert member.owned_experts == []
    assert member.kv_seq == 0
    assert member.status.value == "alive"


def test_routing_table_epoch_order() -> None:
    older = RoutingTable(epoch=1, assignment={})
    newer = RoutingTable(epoch=2, assignment={})
    assert newer.is_newer_than(older)
    assert not older.is_newer_than(newer)


def test_routing_table_tie_break_is_total() -> None:
    a = RoutingTable(epoch=3, assignment={0: RoutingEntry(node_id="n1", slot=0)})
    b = RoutingTable(epoch=3, assignment={0: RoutingEntry(node_id="n2", slot=0)})
    # Same epoch, different assignment: exactly one wins, deterministically.
    assert a.is_newer_than(b) != b.is_newer_than(a)


def test_cluster_view_holds_members_and_routing() -> None:
    view = ClusterView(
        members={"n1": _member("n1")},
        routing=RoutingTable(epoch=0, assignment={}),
    )
    assert view.members["n1"].node_id == "n1"
    assert view.routing is not None and view.routing.epoch == 0
