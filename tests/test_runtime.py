"""Tests for the live runtime state: membership, routing table, aggregate view."""

from mixup_core.election import leader_projection
from mixup_core.runtime import (
    ClusterView,
    LoadMetrics,
    MemberState,
    RoutingEntry,
    RoutingTable,
    apply_routing,
)
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


def test_routing_table_equal_tables_are_not_newer() -> None:
    a = RoutingTable(epoch=3, assignment={0: RoutingEntry(node_id="n1", slot=0)})
    b = RoutingTable(epoch=3, assignment={0: RoutingEntry(node_id="n1", slot=0)})
    assert not a.is_newer_than(b)
    assert not b.is_newer_than(a)


def test_cluster_view_holds_members_and_routing() -> None:
    view = ClusterView(
        members={"n1": _member("n1")},
        routing=RoutingTable(epoch=0, assignment={}),
    )
    assert view.members["n1"].node_id == "n1"
    assert view.routing is not None and view.routing.epoch == 0


def test_apply_routing_adopts_newer_table() -> None:
    view = ClusterView(members={"n1": _member("n1")}, routing=RoutingTable(epoch=1, assignment={}))
    newer = RoutingTable(epoch=2, assignment={})
    assert apply_routing(view, newer).routing is not None
    assert apply_routing(view, newer).routing.epoch == 2


def test_apply_routing_ignores_older_table() -> None:
    view = ClusterView(members={"n1": _member("n1")}, routing=RoutingTable(epoch=2, assignment={}))
    older = RoutingTable(epoch=1, assignment={})
    assert apply_routing(view, older).routing is not None
    assert apply_routing(view, older).routing.epoch == 2


def test_apply_routing_adopts_when_none() -> None:
    view = ClusterView(members={"n1": _member("n1")})
    table = RoutingTable(epoch=1, assignment={})
    assert apply_routing(view, table).routing is not None
    assert apply_routing(view, table).routing.epoch == 1


def test_expert_moves_reports_owner_changes() -> None:
    from mixup_core.runtime import expert_moves

    old = RoutingTable(
        epoch=0,
        assignment={
            0: RoutingEntry(node_id="n1", slot=0),
            1: RoutingEntry(node_id="n1", slot=1),
            2: RoutingEntry(node_id="n2", slot=0),
        },
    )
    new = RoutingTable(
        epoch=1,
        assignment={
            0: RoutingEntry(node_id="n2", slot=1),
            1: RoutingEntry(node_id="n2", slot=2),
            2: RoutingEntry(node_id="n2", slot=0),
        },
    )
    assert expert_moves(old, new) == [(0, "n1", "n2"), (1, "n1", "n2")]


def test_expert_moves_empty_when_no_change() -> None:
    from mixup_core.runtime import expert_moves

    table = RoutingTable(epoch=0, assignment={0: RoutingEntry(node_id="n1", slot=0)})
    assert expert_moves(table, table) == []


def test_leader_projection() -> None:
    view = ClusterView(
        members={
            "leader": MemberState(
                node_id="leader", role=NodeRole.LEADER, epoch=4, incarnation=1, priority=1, kv_seq=9
            ),
            "n1": _member("n1"),
        },
        routing=RoutingTable(epoch=3, assignment={}),
    )
    proj = leader_projection(view)
    assert proj.leader == "leader"
    assert proj.epoch == 4
    assert proj.kv_seq == 9
    assert proj.routing_epoch == 3


def test_leader_projection_empty_view() -> None:
    proj = leader_projection(ClusterView(members={}))
    assert proj.leader is None
    assert proj.epoch == 0
    assert proj.kv_seq == 0
    assert proj.routing_epoch == 0


def test_leader_projection_prefers_current_leader_over_priority() -> None:
    view = ClusterView(
        members={
            "old": MemberState(
                node_id="old", role=NodeRole.EXPERT, epoch=0, incarnation=1, priority=9
            ),
            "cur": MemberState(
                node_id="cur", role=NodeRole.LEADER, epoch=1, incarnation=1, priority=1
            ),
        }
    )
    # "old" has higher static priority but demoted; the projection must report "cur".
    assert leader_projection(view).leader == "cur"
