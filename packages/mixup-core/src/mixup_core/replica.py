"""Replica-aware fault tolerance: assign backups and promote them on failure.

Each expert's routing entry carries a ``backup`` node (a different device). On a primary
failure the backup is promoted to primary, so the loss is absorbed by re-routing to the
backup holder rather than re-sharding the whole pool — the report's §3.8 mechanism.
Experts whose backup is also gone fall back to normal re-placement.
"""

from .gossip import Dead, apply_gossip
from .runtime import ClusterView, RoutingEntry, RoutingTable
from .scheduler import LeastLoadedPolicy, finalize
from .types import ExpertId, MemberStatus, NodeId


def plan_backups(table: RoutingTable, alive: set[NodeId]) -> RoutingTable:
    """Assign each expert a backup on a different alive node, load-balanced.

    Preserves the primary assignment, slots, and epoch; only fills in ``backup``. When only
    one node is alive, backups are ``None`` (no possible different-node backup).
    """
    nodes = sorted(alive)
    backup_load: dict[NodeId, int] = {n: 0 for n in nodes}
    entries: dict[ExpertId, RoutingEntry] = {}
    for expert, entry in sorted(table.assignment.items()):
        candidates = [n for n in nodes if n != entry.node_id]
        backup: NodeId | None = None
        if candidates:
            backup = min(candidates, key=lambda n: (backup_load[n], n))
            backup_load[backup] += 1
        entries[expert] = entry.model_copy(update={"backup": backup})
    return RoutingTable(epoch=table.epoch, assignment=entries)


def promote_backups(
    table: RoutingTable, dead_node: NodeId, alive: set[NodeId]
) -> RoutingTable:
    """Promote backups for ``dead_node``'s experts; bump the epoch and re-assign backups.

    Experts whose backup is also dead (or absent) are dropped from the primary map and must
    be re-placed by the caller via ``rebalance_experts`` (returned as ``orphans``).
    """
    primary: dict[ExpertId, NodeId] = {}
    orphans: list[ExpertId] = []
    for expert, entry in table.assignment.items():
        if entry.node_id == dead_node:
            if entry.backup is not None and entry.backup in alive:
                primary[expert] = entry.backup
            else:
                orphans.append(expert)
        else:
            primary[expert] = entry.node_id

    new_table = finalize(primary, table.epoch + 1)
    return plan_backups(new_table, alive), orphans


def failover_with_backups(
    view: ClusterView,
    dead_node: NodeId,
    by_node: NodeId,
    policy: object | None = None,
) -> ClusterView:
    """Mark ``dead_node`` DEAD, promote its experts' backups, and re-place any orphans.

    The full replica-aware failover: experts whose backup is alive go straight to that
    backup; experts with no surviving backup fall back to ``policy`` (least-loaded by
    default). Returns a new view whose routing (epoch bumped) still covers every expert.
    """
    member = view.members.get(dead_node)
    table = view.routing
    already_dead = member is not None and member.status is MemberStatus.DEAD
    # Idempotent: if the node is already dead and owns no experts, the failover was
    # already applied. Re-probing a dead peer (failure-detection self-healing) must not
    # bump the routing epoch or re-shard the pool again — that churn is what turns a
    # transient stall into an endless rebalance cycle.
    if (
        already_dead
        and table is not None
        and not any(entry.node_id == dead_node for entry in table.assignment.values())
    ):
        return view
    incarnation = member.incarnation if member else 1
    view = apply_gossip(view, Dead(node_id=dead_node, incarnation=incarnation, by_node=by_node))
    if table is None:
        return view

    alive = {m.node_id for m in view.members.values() if m.status == MemberStatus.ALIVE}
    new_table, orphans = promote_backups(table, dead_node, alive)
    if orphans:
        placement = (policy or LeastLoadedPolicy()).place(view, orphans)
        primary = {e: entry.node_id for e, entry in new_table.assignment.items()}
        primary.update(placement)
        new_table = finalize(primary, new_table.epoch + 1)
        new_table = plan_backups(new_table, alive)
    return ClusterView(members=view.members, routing=new_table)
