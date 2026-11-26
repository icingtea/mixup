# Testing & documentation

**Goal:** prove correctness of the distributed logic *without* a GPU or a cluster, and
provide a chaos harness for the integration behavior.

## Why this shape

Distributed-systems bugs are timing/message-order bugs; they're the hardest to catch with
ad-hoc testing. The design therefore keeps **all decision logic torch-free in
`mixup-core`** (Phases 1–3, 6) so it can be exercised by a **deterministic simulator** —
no torch, no Docker, no network. This is the same principle as today's
`tests/test_topology.py` (routing is a pure function), extended to the whole control
plane.

## Torch-free unit tests (the bulk, CI-runnable)

| Module | What is tested |
|---|---|
| `startup_config` | rename + the two new fields' validation |
| `runtime` | `RoutingTable.is_newer_than` totality/transitivity; `ClusterView` merge |
| `gossip` | SWIM state machine transitions: alive→suspect→dead, resurrection via higher incarnation, stale-`Dead` vs newer `Alive` |
| `election` | rank order `(kv_seq, priority, id)`; single winner; stale-epoch yield; symmetric-tie convergence |
| `scheduler` | `rebalance_experts` on death/partition/capacity/no-op; coverage+uniqueness invariants |
| `kvledger` | header encode/decode round-trip; monotonic discard rule |

These run via `pytest` exactly like the current `tests/` (throwaway venv on NixOS, or
`uv run pytest` elsewhere — see `docs/development.md`).

## Deterministic simulation harness

A small in-core simulator drives the pure state machines through scripted scenarios:

```
Simulator:
  - a set of nodes, each running apply_gossip/elect/rebalance as pure functions
  - a message scheduler with controllable drop/delay/reorder/partition
  - a crash/restart injector (incarnation++)
```

Scenarios (each asserts an invariant from [architecture.md](architecture.md)):

1. **Single expert crash** → coverage restored (invariant #2), no double-owner.
2. **Leader crash with ledger lag** → the highest-`kv_seq` survivor wins (invariant #1).
3. **Network partition, then heal** → at most one leader per epoch; loser demotes.
4. **Message reorder storm** → routing table never exposes a half-applied assignment
   (invariant #4).
5. **Crash + restart** → incarnation revival; no zombie membership.

The simulator uses a **deterministic PRNG** seed so failures reproduce exactly. This is
the standard model-checking approach for membership/consensus logic ([SWIM], [Raft]'s own
test methodology).

## Integration / chaos harness (manual + optional CI, needs Docker + GPU)

A `scripts/chaos.sh` + an integration test module drive the real cluster:

- Start the cluster, run a known prompt, capture output (the **no-failure baseline**).
- `docker stop` an expert mid-generation → generation **completes correctly**: the output
  is a valid, terminating completion with no double-count and no crash. Exact token match
  with the baseline is **not** required — rebalancing the experts shifts the float reduce
  grouping (see Phase 7).
- `docker stop` the leader mid-generation → generation completes correctly against the new
  leader (shed + rebalance + rehydration); exact token match is not required for the same
  reason.
- `docker network disconnect`/`connect` to simulate partitions.
- Assert per-node memory stays at the follower baseline except while leader (invariant #5).

This is modeled on **chaos engineering** practice — inject real failures, assert a
steady-state invariant ([ChaosEng]).

## Documentation updates (delivered with the code)

- **New ADRs** in `docs/decisions.md`: gossip/SWIM membership, freshness-aware election,
  KV-cache ledger, lazy role loading, run-based dynamic placement, HTTP-over-gRPC
  transport, persistent leader (closing ADR-011).
- **Rewrite** `docs/config.md` (StartupConfig schema), `docs/protocol.md` (gossip +
  ledger endpoints), `docs/math.md` (ledger/rebalance/rehydration math), `docs/running.md`
  (failover drills), `.env.example` (all tunables).
- **This plan** (`docs/plan/`) remains as the design record and is linked from
  `docs/README.md`.

## Definition of done

- All torch-free unit tests + the deterministic simulator pass in CI.
- The chaos harness produces a passing manual run for expert-failover and leader-failover.
- `ruff check .` clean; docs updated; `mixup-smoketest` still passes against the renamed
  `startup_config.json`.
