# Phase 0 — Rename `Config` → `StartupConfig`

**Goal:** rename the static layout type and its file/function, with zero behavior
change. This creates the vocabulary needed by every later phase: `StartupConfig` is the
*initial* layout, distinct from the live `RoutingTable` introduced in Phase 1.

**Status:** independent of all other phases; land as its own commit first.

## Why this name

Today the single `Config` type is overloaded: it means "the whole cluster layout," but
after this project the cluster layout *changes at runtime* (failover, rebalancing). The
word `StartupConfig` makes explicit that this document describes only the **initial**
placement and initial leader. Everything that follows in Phases 1–6 mutates a *different*
object (the `RoutingTable` / `ClusterView`), never `StartupConfig`.

This matters for correctness: reusing `Config` for runtime state would invite code that
mutates the startup document, breaking the "config is the single source of truth for the
*boot* layout" property. Keeping them as separate types makes the boundary mechanical.

## What changes

| From | To |
|---|---|
| `mixup_core.types.Config` | `mixup_core.types.StartupConfig` |
| `mixup_core.config.load_config_file` | `mixup_core.startup_config.load_startup_config` |
| `packages/mixup-core/src/mixup_core/config.py` | `.../startup_config.py` |
| `config.json` | `startup_config.json` |
| `distribute.py:DEFAULT_CONFIG_PATH = "config.json"` | `"startup_config.json"` |
| `smoketest.py` default `config` arg | `"startup_config.json"` |

### New `StartupConfig` shape

Beyond the rename, add two fields that later phases need but that don't change current
behavior:

```json
{
  "initial_leader": "container_1",
  "nodes": ["container_1", "container_2", "container_3", "container_4", "container_5"],
  "experts": {
    "container_2": [0, 1, 2, 3, 4, 5, 6, 7],
    "container_3": [8, 9, 10, 11, 12, 13, 14, 15],
    "container_4": [16, 17, 18, 19, 20, 21, 22, 23],
    "container_5": [24, 25, 26, 27, 28, 29, 30, 31]
  }
}
```

- `initial_leader` replaces the old `central` field (same meaning, clearer name: the
  leader *starts* here, but may move after an election).
- `nodes` is the full membership roster — needed by gossip/election so every node knows
  who its peers are at boot (gossip then takes over dynamically). `nodes[0]` can be the
  implicit highest priority, but we introduce an explicit `priority` per node in Phase 1
  rather than overloading ordering.
- `experts` is the **initial** assignment only.

The existing validators are preserved (renamed only):
- **uniqueness** — an expert id assigned to at most one node;
- **leader-not-an-expert-holder** — `initial_leader ∉ experts` (the leader dispatches,
  holds no experts).

`role(name)` becomes `initial_role(name)` returning `LEADER`/`EXPERT` (`NodeRole`).

## Methodology

This is a mechanical, compiler-guided rename: rename the symbol, run `ruff` and the
test suite, and let the import errors point at every remaining reference. The
discipline is **no behavior change in this phase** — the diff should be renames and the
two new (unused yet) fields only. This keeps the rename reviewable in isolation before
any distributed-system logic lands.

## Concrete changes

- `packages/mixup-core/src/mixup_core/types.py` — rename class, add fields, rename
  validators' messages.
- `packages/mixup-core/src/mixup_core/config.py` → `startup_config.py` — rename function
  to `load_startup_config`.
- `packages/mixup-core/src/mixup_core/__init__.py` — update imports/`__all__`.
- `packages/mixup-core/src/mixup_core/env.py` — `NodeEnv.config: StartupConfig`.
- `packages/mixup-node/src/mixup_node/{infer,dispatch}.py` — update imports.
- `packages/mixup-orchestrator/src/mixup_orchestrator/distribute.py` — update import,
  `DEFAULT_CONFIG_PATH`, and the field name used for the leader.
- `tests/test_config.py` — rename + extend for the two new fields.
- `config.json` → `startup_config.json` (git `mv`).
- `docs/config.md`, `docs/decisions.md`, `.env.example` — update references.

## Acceptance criteria

- `ruff check .` and `pytest` pass.
- `mixup-smoketest startup_config.json` still reports a valid, contiguous partition.
- The diff contains no behavioral change (new fields are parsed but not yet read).

## Open questions

- Keep the filename `config.json` for backward compatibility of any scripts, or rename
  outright? **Decision (default):** rename outright; it's a prototype with no external
  users.
- Should `nodes` be explicit, or derived from `experts` + `initial_leader`? **Default:**
  explicit, because Phase 1 assigns per-node priorities and future clusters may have
  idle/standby nodes that hold no experts at boot.
