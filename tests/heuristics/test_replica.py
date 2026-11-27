"""Tests for replica-aware failover: backup assignment and promotion."""

from mixup_core.replica import plan_backups, promote_backups
from mixup_core.runtime import RoutingEntry, RoutingTable


def _table(blocks: dict[str, tuple[int, int]]) -> RoutingTable:
    assignment: dict[int, RoutingEntry] = {}
    for node, (lo, hi) in blocks.items():
        for slot, expert in enumerate(range(lo, hi + 1)):
            assignment[expert] = RoutingEntry(node_id=node, slot=slot)
    return RoutingTable(epoch=1, assignment=assignment)


def test_plan_backups_assigns_different_node() -> None:
    table = plan_backups(_table({"a": (0, 3), "b": (4, 7)}), {"a", "b"})
    for _expert, entry in table.assignment.items():
        assert entry.backup is not None
        assert entry.backup != entry.node_id


def test_plan_backups_balances_load() -> None:
    table = plan_backups(_table({"a": (0, 7)}), {"a", "b", "c"})
    # a owns 8 experts; backups spread across b and c.
    backups = [e.backup for e in table.assignment.values()]
    assert set(backups) == {"b", "c"}
    assert backups.count("b") == 4 and backups.count("c") == 4


def test_plan_backups_single_node_is_none() -> None:
    table = plan_backups(_table({"a": (0, 3)}), {"a"})
    assert all(e.backup is None for e in table.assignment.values())


def test_promote_backups_promotes_and_reassigns() -> None:
    table = plan_backups(_table({"a": (0, 3), "b": (4, 7)}), {"a", "b"})
    new_table, orphans = promote_backups(table, "a", {"b"})
    # a's experts (0-3) went to b (their backup); no orphans.
    assert orphans == []
    assert all(new_table.assignment[e].node_id == "b" for e in range(8))
    assert new_table.epoch == table.epoch + 1


def test_promote_backups_orphans_when_backup_dead() -> None:
    table = plan_backups(_table({"a": (0, 3), "b": (4, 7)}), {"a", "b"})
    # both a and b dead -> a's experts have no alive backup (orphans).
    new_table, orphans = promote_backups(table, "a", {"c"})
    assert set(orphans) == {0, 1, 2, 3}
    # b's experts (4-7) survive on b... but b is not in alive, so they're dropped too?
    # b is not dead here, only a is. b's experts stay on b.
    assert set(new_table.assignment) == {4, 5, 6, 7}


def test_promote_backups_keeps_survivors() -> None:
    table = plan_backups(_table({"a": (0, 3), "b": (4, 7), "c": (8, 11)}), {"a", "b", "c"})
    new_table, orphans = promote_backups(table, "b", {"a", "c"})
    # b's experts (4-7) promoted to their backups; a and c keep theirs.
    assert orphans == []
    assert set(new_table.assignment) == set(range(12))
    assert new_table.assignment[0].node_id == "a"
