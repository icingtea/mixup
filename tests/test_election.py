"""Tests for freshness-aware leader election."""

from mixup_core.election import elect
from mixup_core.runtime import ClusterView, MemberState
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    priority: int = 1,
    kv_seq: int = 0,
    status: MemberStatus = MemberStatus.ALIVE,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        status=status,
        epoch=0,
        incarnation=1,
        priority=priority,
        kv_seq=kv_seq,
    )


def _view(*members: MemberState) -> ClusterView:
    return ClusterView(members={m.node_id: m for m in members})


def test_elect_prefers_freshest_kv_seq() -> None:
    view = _view(_member("a", priority=10, kv_seq=2), _member("b", priority=1, kv_seq=9))
    assert elect(view) == "b"


def test_elect_tie_breaks_by_priority_then_id() -> None:
    view = _view(_member("b", priority=5, kv_seq=3), _member("a", priority=5, kv_seq=3))
    assert elect(view) == "b"  # same kv_seq, same priority -> higher node_id


def test_elect_ignores_dead() -> None:
    view = _view(
        _member("a", priority=10, kv_seq=5, status=MemberStatus.DEAD),
        _member("b", priority=1, kv_seq=1),
    )
    assert elect(view) == "b"


def test_elect_none_when_all_dead() -> None:
    view = _view(_member("a", status=MemberStatus.DEAD))
    assert elect(view) is None
