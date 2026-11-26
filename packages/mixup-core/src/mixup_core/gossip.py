"""SWIM membership state machine (pure logic; transport lives in mixup-node).

This is the failure-detection core of Phase 2, written as a pure function so it can be
table-tested with no network. ``apply_gossip`` maps ``(view, message) -> view``: it only
updates membership *status* (alive / suspect / dead); the transport decides *when* to send
pings and, on timeout, emits a ``Suspect``.

References: [SWIM], [Serf-gossip].
"""

from pydantic import BaseModel

from .runtime import ClusterView
from .types import Epoch, Incarnation, MemberStatus, NodeId


class Ping(BaseModel):
    """Direct probe from the sender to a target (node_id is the sender)."""

    node_id: NodeId
    incarnation: Incarnation
    epoch: Epoch
    kv_seq: int


class Ack(BaseModel):
    """Reply to a ping; node_id is the node being acked."""

    node_id: NodeId
    incarnation: Incarnation
    epoch: Epoch
    kv_seq: int
    routing_epoch: Epoch = 0


class PingReq(BaseModel):
    """Ask a peer to indirectly probe `target` on our behalf."""

    from_node: NodeId
    target: NodeId
    incarnation: Incarnation


class Suspect(BaseModel):
    """Declare `node_id` suspect (direct + indirect probes failed)."""

    node_id: NodeId
    incarnation: Incarnation
    by_node: NodeId


class Alive(BaseModel):
    """Declare `node_id` alive; a higher incarnation revives a suspect/dead member."""

    node_id: NodeId
    incarnation: Incarnation
    epoch: Epoch


class Dead(BaseModel):
    """Declare `node_id` dead."""

    node_id: NodeId
    incarnation: Incarnation
    by_node: NodeId


GossipMsg = Ping | Ack | PingReq | Suspect | Alive | Dead
"""Any message the SWIM state machine accepts."""


def _set_status(view: ClusterView, node_id: NodeId, status: MemberStatus) -> ClusterView:
    member = view.members[node_id]
    updated = member.model_copy(update={"status": status})
    return ClusterView(members={**view.members, node_id: updated}, routing=view.routing)


def apply_gossip(view: ClusterView, msg: GossipMsg) -> ClusterView:
    """Return a new view with membership status updated by ``msg``.

    Rules:
    - ``Ping``/``Ack``/``PingReq`` don't change status (transport-level liveness).
    - A message with ``incarnation < member.incarnation`` is stale and ignored.
    - ``Alive`` revives any status to ``ALIVE`` (resurrection).
    - ``Suspect`` downgrades ``ALIVE`` -> ``SUSPECT`` only.
    - ``Dead`` downgrades to ``DEAD``.
    """
    if isinstance(msg, (Ping, Ack, PingReq)):
        return view

    member = view.members.get(msg.node_id)
    if member is None:
        # Membership is seeded at boot from StartupConfig; we don't adopt unknown
        # members from a bare status message (the transport handles joins).
        return view
    if msg.incarnation < member.incarnation:
        return view

    if isinstance(msg, Alive):
        return _set_status(view, msg.node_id, MemberStatus.ALIVE)
    if isinstance(msg, Suspect):
        if member.status is MemberStatus.ALIVE:
            return _set_status(view, msg.node_id, MemberStatus.SUSPECT)
        return view
    if isinstance(msg, Dead):
        return _set_status(view, msg.node_id, MemberStatus.DEAD)
    return view
