"""Tests for the SWIM membership state machine (apply_gossip)."""

from mixup_core.gossip import Alive, Dead, Ping, Suspect, apply_gossip
from mixup_core.runtime import ClusterView, MemberState
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    incarnation: int = 1,
    status: MemberStatus = MemberStatus.ALIVE,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        status=status,
        epoch=0,
        incarnation=incarnation,
        priority=1,
    )


def _view(*members: MemberState) -> ClusterView:
    return ClusterView(members={m.node_id: m for m in members})


def test_suspect_marks_alive_suspect() -> None:
    out = apply_gossip(_view(_member("a")), Suspect(node_id="a", incarnation=1, by_node="b"))
    assert out.members["a"].status is MemberStatus.SUSPECT


def test_dead_marks_dead() -> None:
    view = _view(_member("a", status=MemberStatus.SUSPECT))
    out = apply_gossip(view, Dead(node_id="a", incarnation=1, by_node="b"))
    assert out.members["a"].status is MemberStatus.DEAD


def test_alive_resurrects_dead_member() -> None:
    view = _view(_member("a", incarnation=2, status=MemberStatus.DEAD))
    out = apply_gossip(view, Alive(node_id="a", incarnation=2, epoch=0))
    assert out.members["a"].status is MemberStatus.ALIVE


def test_stale_message_ignored() -> None:
    view = _view(_member("a", incarnation=5, status=MemberStatus.SUSPECT))
    out = apply_gossip(view, Dead(node_id="a", incarnation=4, by_node="b"))
    assert out.members["a"].status is MemberStatus.SUSPECT


def test_suspect_does_not_revive_dead() -> None:
    view = _view(_member("a", status=MemberStatus.DEAD))
    out = apply_gossip(view, Suspect(node_id="a", incarnation=1, by_node="b"))
    assert out.members["a"].status is MemberStatus.DEAD


def test_ping_is_a_noop() -> None:
    view = _view(_member("a"))
    out = apply_gossip(view, Ping(node_id="a", incarnation=1, epoch=0, kv_seq=0))
    assert out.members["a"].status is MemberStatus.ALIVE


def test_unknown_member_ignored() -> None:
    view = _view(_member("a"))
    out = apply_gossip(view, Dead(node_id="zzz", incarnation=1, by_node="b"))
    assert "zzz" not in out.members
