"""Tests for freshness-aware leader election."""

from mixup_core.election import (
    apply_leader_announcement,
    current_leader,
    elect,
    promote,
    should_become_leader,
)
from mixup_core.runtime import ClusterView, MemberState
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    priority: int = 1,
    kv_seq: int = 0,
    status: MemberStatus = MemberStatus.ALIVE,
    role: NodeRole = NodeRole.EXPERT,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=role,
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


def test_current_leader_returns_alive_leader() -> None:
    view = _view(_member("l", role=NodeRole.LEADER, priority=3), _member("a", priority=2))
    assert current_leader(view) == "l"


def test_current_leader_none_when_leader_dead() -> None:
    view = _view(
        _member("l", role=NodeRole.LEADER, status=MemberStatus.DEAD, priority=3),
        _member("a", priority=2),
    )
    assert current_leader(view) is None


def test_should_become_leader_true_when_winner() -> None:
    # leader dead; "a" is the highest-priority alive node.
    view = _view(
        _member("l", role=NodeRole.LEADER, status=MemberStatus.DEAD, priority=5),
        _member("a", priority=3),
        _member("b", priority=2),
    )
    assert should_become_leader(view, "a") is True
    assert should_become_leader(view, "b") is False


def test_should_become_leader_false_when_leader_alive() -> None:
    view = _view(_member("l", role=NodeRole.LEADER, priority=5), _member("a", priority=3))
    assert should_become_leader(view, "a") is False


def test_promote_sets_role_and_demotes_old_leader() -> None:
    view = _view(
        _member("l", role=NodeRole.LEADER, status=MemberStatus.DEAD, priority=5, kv_seq=3),
        _member("a", priority=3),
    )
    out = promote(view, "a")
    assert out.members["a"].role is NodeRole.LEADER
    assert out.members["a"].epoch == 1  # bumped
    assert out.members["l"].role is NodeRole.EXPERT  # demoted


def test_promote_bumps_epoch_past_max() -> None:
    view = _view(
        _member("l", role=NodeRole.LEADER, status=MemberStatus.DEAD, priority=5),
        _member("a", priority=3),
    )
    # give l a high epoch so the new epoch must exceed it
    view = ClusterView(
        members={
            "l": MemberState(
                node_id="l", role=NodeRole.LEADER, status=MemberStatus.DEAD, epoch=7,
                incarnation=1, priority=5,
            ),
            "a": _member("a", priority=3),
        }
    )
    out = promote(view, "a")
    assert out.members["a"].epoch == 8


def test_apply_leader_announcement_adopts_and_demotes() -> None:
    view = _view(
        _member("l", role=NodeRole.LEADER, status=MemberStatus.DEAD, priority=5),
        _member("a", priority=3),
    )
    out = apply_leader_announcement(view, "a", epoch=3)
    assert out.members["a"].role is NodeRole.LEADER
    assert out.members["a"].epoch == 3
    assert out.members["a"].status is MemberStatus.ALIVE
    assert out.members["l"].role is NodeRole.EXPERT  # demoted


def test_apply_leader_announcement_ignores_stale_epoch() -> None:
    view = ClusterView(
        members={
            "l": MemberState(
                node_id="l", role=NodeRole.LEADER, epoch=7, incarnation=1, priority=5
            ),
            "a": _member("a", priority=3),
        }
    )
    out = apply_leader_announcement(view, "a", epoch=3)
    assert out.members["l"].role is NodeRole.LEADER  # unchanged
    assert out.members["a"].role is NodeRole.EXPERT
