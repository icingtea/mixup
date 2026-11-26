"""Freshness-aware leader election (pure function).

Picks the highest-ranked *alive* member, ranked by ``(kv_seq, priority, node_id)``. The
``kv_seq``-first ordering makes the election prefer the node with the most up-to-date
KV-cache ledger, so the winner can resume generation with the least recompute — the same
"most up-to-date log wins" idea as Raft's election restriction ([Raft] §5.4.1).
"""

from .runtime import ClusterView, LeaderProjection
from .types import MemberStatus, NodeId


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
