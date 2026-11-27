"""SWIM membership state machine (pure logic; transport lives in mixup-node).

This is the failure-detection core of Phase 2, written as a pure function so it can be
table-tested with no network. ``apply_gossip`` maps ``(view, message) -> view``: it only
updates membership *status* (alive / suspect / dead); the transport decides *when* to send
pings and, on timeout, emits a ``Suspect``.

References: [SWIM], [Serf-gossip].
"""

from pydantic import BaseModel

from .runtime import ClusterView, MemberState
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
        if member.status == MemberStatus.ALIVE:
            return _set_status(view, msg.node_id, MemberStatus.SUSPECT)
        return view
    if isinstance(msg, Dead):
        return _set_status(view, msg.node_id, MemberStatus.DEAD)
    return view


_STATUS_RANK = {
    MemberStatus.ALIVE: 0,
    MemberStatus.SUSPECT: 1,
    MemberStatus.DEAD: 2,
}


def _member_newer(a: MemberState, b: MemberState) -> bool:
    """Is `a`'s membership state newer than `b`'s?

    A higher incarnation always wins (a restart). Within the same incarnation, status
    progresses ALIVE -> SUSPECT -> DEAD, so the more-progressed status wins.
    """
    if a.incarnation != b.incarnation:
        return a.incarnation > b.incarnation
    return _STATUS_RANK[MemberStatus(a.status)] > _STATUS_RANK[MemberStatus(b.status)]


def merge_views(local: ClusterView, remote: ClusterView) -> ClusterView:
    """Anti-entropy reconciliation: adopt the newer of each member's state and the newer table.

    Used by push-pull gossip to converge two nodes' views. For each member present in both
    views, the newer state wins (see ``_member_newer``); members in only one view are kept,
    and the routing table is adopted only if the remote's is newer.
    """
    members = dict(local.members)
    for node_id, remote_member in remote.members.items():
        local_member = members.get(node_id)
        if local_member is None or _member_newer(remote_member, local_member):
            members[node_id] = remote_member

    routing = local.routing
    if remote.routing is not None and (
        routing is None or remote.routing.is_newer_than(routing)
    ):
        routing = remote.routing

    return ClusterView(members=members, routing=routing)


_TYPE_BY_CLASS = {
    Ping: "ping",
    Ack: "ack",
    PingReq: "pingreq",
    Suspect: "suspect",
    Alive: "alive",
    Dead: "dead",
}
_CLASS_BY_TYPE = {v: k for k, v in _TYPE_BY_CLASS.items()}


def encode_gossip(msg: GossipMsg) -> dict:
    """Serialize a gossip message to a JSON-ready dict with a ``type`` discriminator.

    The discriminator is required because ``Suspect`` and ``Dead`` share the same fields.
    """
    return {"type": _TYPE_BY_CLASS[type(msg)], **msg.model_dump(mode="json")}


def decode_gossip(data: dict) -> GossipMsg:
    """Deserialize a gossip message from a dict produced by ``encode_gossip``."""
    cls = _CLASS_BY_TYPE[data["type"]]
    return cls(**{k: v for k, v in data.items() if k != "type"})
