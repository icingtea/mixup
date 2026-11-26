"""Tests for anti-entropy view reconciliation (gossip merge)."""

from mixup_core.gossip import merge_views
from mixup_core.runtime import ClusterView, MemberState, RoutingEntry, RoutingTable
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    incarnation: int = 1,
    status: MemberStatus = MemberStatus.ALIVE,
    kv_seq: int = 0,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        status=status,
        epoch=0,
        incarnation=incarnation,
        priority=1,
        kv_seq=kv_seq,
    )


def _view(*members: MemberState, routing: RoutingTable | None = None) -> ClusterView:
    return ClusterView(members={m.node_id: m for m in members}, routing=routing)


def test_newer_incarnation_wins_even_over_dead() -> None:
    local = _view(_member("a", incarnation=1, status=MemberStatus.DEAD))
    remote = _view(_member("a", incarnation=2, status=MemberStatus.ALIVE))
    merged = merge_views(local, remote)
    assert merged.members["a"].status is MemberStatus.ALIVE
    assert merged.members["a"].incarnation == 2


def test_same_incarnation_dead_overrides_alive() -> None:
    local = _view(_member("a", status=MemberStatus.ALIVE))
    remote = _view(_member("a", status=MemberStatus.DEAD))
    assert merge_views(local, remote).members["a"].status is MemberStatus.DEAD


def test_same_incarnation_suspect_overrides_alive() -> None:
    local = _view(_member("a", status=MemberStatus.ALIVE))
    remote = _view(_member("a", status=MemberStatus.SUSPECT))
    assert merge_views(local, remote).members["a"].status is MemberStatus.SUSPECT


def test_remote_only_members_are_added() -> None:
    local = _view(_member("a"))
    remote = _view(_member("a"), _member("b"))
    assert "b" in merge_views(local, remote).members


def test_local_only_members_are_kept() -> None:
    local = _view(_member("a"), _member("b"))
    remote = _view(_member("a"))
    assert "b" in merge_views(local, remote).members


def test_newer_routing_table_wins() -> None:
    old = RoutingTable(epoch=1, assignment={0: RoutingEntry(node_id="a", slot=0)})
    new = RoutingTable(epoch=2, assignment={0: RoutingEntry(node_id="b", slot=0)})
    local = _view(_member("a"), routing=old)
    remote = _view(_member("a"), routing=new)
    assert merge_views(local, remote).routing is not None
    assert merge_views(local, remote).routing.epoch == 2


def test_local_routing_kept_when_remote_is_older() -> None:
    old = RoutingTable(epoch=1, assignment={})
    new = RoutingTable(epoch=2, assignment={})
    local = _view(_member("a"), routing=new)
    remote = _view(_member("a"), routing=old)
    assert merge_views(local, remote).routing is not None
    assert merge_views(local, remote).routing.epoch == 2
