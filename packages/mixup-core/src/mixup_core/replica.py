"""Replica-aware fault tolerance: assign backups and promote them on failure.

Each expert's routing entry carries a ``backup`` node (a different device). On a primary
failure the backup is promoted to primary, so the loss is absorbed by re-routing to the
backup holder rather than re-sharding the whole pool — the report's §3.8 mechanism.
Experts whose backup is also gone fall back to normal re-placement.
"""

from .runtime import RoutingEntry, RoutingTable
from .scheduler import finalize
from .types import ExpertId, NodeId


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
