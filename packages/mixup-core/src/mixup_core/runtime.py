"""Live cluster state: membership, the versioned routing table, and the aggregate view.

Unlike ``StartupConfig`` (the fixed boot layout), these types represent the cluster as
it changes at runtime. All of it is pure pydantic — no torch, no sockets — so the
scheduler, election, and gossip state machine can be unit-tested on the host.
"""

from pydantic import BaseModel

from .types import Epoch, ExpertId, Incarnation, MemberStatus, NodeId, NodePriority, NodeRole


class LoadMetrics(BaseModel):
    """A node's reported load, used by the placement policy (Phase 6)."""

    active_requests: int = 0
    """In-flight expert requests."""

    queue_depth: int = 0
    """Queued but not yet started."""

    free_slots: int = 0
    """Expert capacity headroom."""

    memory_headroom_bytes: int = 0
    """Reported free memory."""


class MemberState(BaseModel):
    """Everything a node knows about one member (itself or a peer)."""

    node_id: NodeId
    role: NodeRole
    status: MemberStatus = MemberStatus.ALIVE
    epoch: Epoch
    incarnation: Incarnation
    priority: NodePriority
    owned_experts: list[ExpertId] = []
    loads: LoadMetrics = LoadMetrics()
    kv_seq: int = 0
    """The freshest KV-cache ledger entry this member holds (election credential)."""


class RoutingEntry(BaseModel):
    """Where one expert lives in the live routing table."""

    node_id: NodeId
    slot: int
    """The expert's local slot index on that node."""


class RoutingTable(BaseModel):
    """The versioned ``ExpertId -> RoutingEntry`` map (the live LUT).

    ``epoch`` makes a stale table unable to overwrite a newer one. The ordering is
    total: epoch first, then a deterministic tie-break over the assignment, so two
    nodes with the same epoch always agree on which table wins.
    """

    epoch: Epoch
    assignment: dict[ExpertId, RoutingEntry]

    def is_newer_than(self, other: "RoutingTable") -> bool:
        """True if ``self`` should replace ``other`` (higher epoch, or deterministic tie-break)."""
        if self.epoch != other.epoch:
            return self.epoch > other.epoch
        return self._canonical() > other._canonical()

    def _canonical(self) -> str:
        """A deterministic, order-independent serialization used only as a tie-break."""
        items = (f"{e}:{r.node_id}:{r.slot}" for e, r in sorted(self.assignment.items()))
        return ",".join(items)


class ClusterView(BaseModel):
    """A node's aggregate view of the cluster: membership + current routing table."""

    members: dict[NodeId, MemberState]
    routing: RoutingTable | None = None


class LeaderProjection(BaseModel):
    """What ``GET /leader`` returns: a read-only projection of the current view."""

    leader: NodeId | None
    epoch: Epoch
    kv_seq: int
    routing_epoch: Epoch


def apply_routing(view: ClusterView, table: RoutingTable) -> ClusterView:
    """Adopt ``table`` if it is newer than the view's current routing table."""
    if view.routing is None or table.is_newer_than(view.routing):
        return ClusterView(members=view.members, routing=table)
    return view


def expert_moves(
    old: RoutingTable,
    new: RoutingTable,
) -> list[tuple[ExpertId, NodeId | None, NodeId]]:
    """Experts whose owner changed between two tables, as ``(expert, old_owner, new_owner)``.

    ``old_owner`` is ``None`` if the expert was not in ``old`` (shouldn't happen on
    failover, but handled defensively).
    """
    moves: list[tuple[ExpertId, NodeId | None, NodeId]] = []
    for expert, entry in new.assignment.items():
        old_entry = old.assignment.get(expert)
        old_owner = old_entry.node_id if old_entry else None
        if old_entry is None or old_entry.node_id != entry.node_id:
            moves.append((expert, old_owner, entry.node_id))
    return moves
