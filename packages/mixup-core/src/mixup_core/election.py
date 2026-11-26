"""Freshness-aware leader election (pure function).

Picks the highest-ranked *alive* member, ranked by ``(kv_seq, priority, node_id)``. The
``kv_seq``-first ordering makes the election prefer the node with the most up-to-date
KV-cache ledger, so the winner can resume generation with the least recompute — the same
"most up-to-date log wins" idea as Raft's election restriction ([Raft] §5.4.1).
"""

from .runtime import ClusterView, LeaderProjection
from .types import MemberStatus, NodeId, NodeRole


def elect(view: ClusterView) -> NodeId | None:
    """Return the node that should be leader, or ``None`` if no member is alive.

    The rank ``(kv_seq, priority, node_id)`` is a total order, so the winner is
    deterministic even when two nodes disagree on message order.
    """
    alive = [m for m in view.members.values() if m.status is MemberStatus.ALIVE]
    if not alive:
        return None
    return max(alive, key=lambda m: (m.kv_seq, m.priority, m.node_id)).node_id


def leader_projection(view: ClusterView) -> LeaderProjection:
    """Project the view into the fields a client needs to discover the leader.

    ``leader`` is the highest-ranked alive member (see ``elect``); the rest are the
    current maximums across the view, so a client can compare epochs/kv_seq between nodes.
    """
    members = view.members.values()
    return LeaderProjection(
        leader=elect(view),
        epoch=max((m.epoch for m in members), default=0),
        kv_seq=max((m.kv_seq for m in members), default=0),
        routing_epoch=view.routing.epoch if view.routing else 0,
    )


def current_leader(view: ClusterView) -> NodeId | None:
    """The alive node with role ``LEADER``, else ``None``."""
    for member in view.members.values():
        if member.role is NodeRole.LEADER and member.status is MemberStatus.ALIVE:
            return member.node_id
    return None


def should_become_leader(view: ClusterView, node_id: NodeId) -> bool:
    """True if there is no alive leader and ``node_id`` is the elected winner.

    Because the rank is a total order, every node with the same view computes the same
    winner, so the winner simply promotes and everyone else waits — no claim/ack needed.
    """
    return current_leader(view) is None and elect(view) == node_id


def promote(view: ClusterView, node_id: NodeId) -> ClusterView:
    """Make ``node_id`` the leader (role ``LEADER``, epoch bumped); demote any other leader.

    The new epoch is ``max(member epochs) + 1``, so a resurrected old leader sees the
    higher epoch and yields (split-brain bound).
    """
    new_epoch = max((m.epoch for m in view.members.values()), default=0) + 1
    members = {
        name: (
            member.model_copy(update={"role": NodeRole.LEADER, "epoch": new_epoch})
            if name == node_id
            else member.model_copy(update={"role": NodeRole.EXPERT})
            if member.role is NodeRole.LEADER
            else member
        )
        for name, member in view.members.items()
    }
    return ClusterView(members=members, routing=view.routing)
