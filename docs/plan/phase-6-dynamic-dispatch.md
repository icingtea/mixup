# Phase 6 — Dynamic expert dispatch (heuristics)

**Goal:** make expert placement a *policy*, not a fixed partition. A pluggable scheduler
recomputes the assignment from live metrics, rebalances incrementally, and failover
becomes just one heuristic ("this node is dead, re-place its experts").

## From static LUT to a live, versioned table

Today the LUT is built once from `StartupConfig.experts` (`build_lut`). This phase
replaces it with the `RoutingTable` (Phase 1) owned by the scheduler:

```
                        ┌───────────────────────────────┐
gossip (loads, loads) ─►│  PlacementPolicy.place(view)  │──► RoutingTable(epoch+1)
                        └───────────────────────────────┘──► gossiped out
```

- Every node runs the **same** pure `rebalance_experts(view, prev, policy)` in core, so
  given the *same* `ClusterView` they reach the *same* table. But views are only
  **eventually consistent**, so two nodes can transiently compute different tables with no
  malfunction; the `is_newer_than`/epoch rule reconciles them. The routing table is **not**
  a CRDT — it is a deterministic function of a non-identical input (see below).
- The table carries an `epoch`; a node applies a table only if it is newer (Phase 1
  `is_newer_than`), and **swaps it atomically** so in-flight requests keep using the old
  table until they complete (RCU-style read-copy-update: publish a new immutable table,
  retire the old one when no request references it — in practice, a reference count or a
  small grace window).

## Run-based expert ownership

A node may own several **runs** (contiguous expert-id ranges), not necessarily one block:

- `ExpertGroup` changes from `experts = experts_for(lut, name)` (one range) to
  `runs: list[(start, end)]` (a set of ranges).
- `_load_layer` loads each run as a `get_slice` and concatenates into a local
  `GraniteMoeExperts` sized to the **owned count**; the global→local slot mapping is
  rebuilt per routing-table version.
- This relaxes the ADR-004 contiguity constraint from "one slice per node" to "one slice
  per run" — which is what makes *incremental* rebalancing possible (a surviving node
  keeps its runs and only adds the orphaned ones).

This matches the *expert-parallel* decomposition used by DeepSpeed-MoE / Tutel /
FasterMoE, where experts are placed in independent groups and routed dynamically
([DeepSpeed-MoE], [Tutel], [FasterMoE]).

## Rebalance algorithm (incremental)

Given `prev` and a changed `view` (a node died, a follower was promoted and shed its
experts, or loads drifted past a threshold):

1. **Unowned experts** = experts whose owner is `DEAD` (or whose owner is over capacity).
2. **Assign** them via `policy.place` across the alive members, respecting each node's
   `free_slots`/capacity and load.
3. **Emit** the new table (bump epoch); each node computes `added = new − old`,
   `removed = old − new`, loads added runs, drops removed runs, and swaps the table.
4. **Dispatch** (Phase 7) immediately routes to the new owner.

`rebalance_experts` is pure and unit-tested against: single-node death, whole-partition
death, capacity overflow, and no-op (loads drifted but nothing moved) cases.

## Placement policies (the "heuristics")

A `PlacementPolicy` is a `Protocol` with `place(view, experts) -> assignment`. Ships with:

| Policy | Heuristic | Use |
|---|---|---|
| `FailoverPolicy` | re-place orphaned experts onto the **least-loaded** survivor | v1 default — this *is* failover |
| `LeastLoadedPolicy` | continuous balancing by `active_requests`/`queue_depth` | proactive rebalance |
| `LatencyAwarePolicy` | weight by measured per-node round-trip latency | v2, needs latency metrics in gossip |
| `CapacityAwarePolicy` | respect `memory_headroom_bytes` / expert capacity | heterogeneous nodes |

Metrics already flow in gossip (`LoadMetrics`, Phase 1/2); policies just consume them.
The scheduler is deliberately **swappable via config** (`startup_config.policy`), so
"complete dynamic dispatch based on heuristics" is an *evolution path*, not a big-bang.

## The `mine`/`slot` masking, revisited

The existing double-masking (zero the weight + `SKIP=999` slot, see `docs/math.md` and
ADR-009) is retained but its inputs become dynamic:

- `mine[e]` and `slot[e]` are rebuilt from the **current** `RoutingTable` (not the boot
  LUT), and cached per table version so the per-request cost stays O(1).
- The `SKIP` sentinel and `_is_expert_parallel`/`grouped_mm` hooks are unchanged (still
  the transformers expert-parallel fast path, [HF-llm-optimization]).

## Consistency of the swap (and the wire epoch)

A request must never hit a node that no longer owns the expert it was sent for, nor be
double-counted. Two rules make that exact:

1. **The request carries the table epoch.** `ExpertRequest` gains a `routing_epoch`
   field (Phase 1's `RoutingTable.epoch`).
2. **Follower serves per the request's epoch.** A follower keeps a short history of the
   last few tables; it applies the `mine`/`slot` mask for the epoch in the request. If it
   no longer holds that epoch, it rejects with the current epoch, and the leader re-issues
   against the new owner (Phase 7).

The atomic versioned swap then guarantees: within one table version, each expert has
exactly one owner (invariant #4). The dispatch side (Phase 7) pins the table version it
used for a request and, on failure, retries against the *new* version.

## Methodology notes

- **Expert parallelism & dynamic placement** are active research areas; we borrow the
  *decomposition* (independent expert groups, dynamic mapping) not the training-side
  machinery. References: [DeepSpeed-MoE], [Tutel], [FasterMoE], and the original sparse
  MoE work [Switch], [GShard], [Mixtral], [Shazeer-MoE].
- **Load balancing** follows the *capacity factor* idea from [GShard]/[Switch]: each node
  has a capacity; overflow experts spill to under-loaded nodes — this is the concrete
  meaning of `free_slots` in `LoadMetrics`.
- **Convergence, not a CRDT.** The rebalance is a deterministic *function*, but because
  no two nodes share an identical view at the same instant, convergence comes from the
  epoch rule (newer wins), not from commutativity. A true CRDT would require a merge of
  divergent states; we do not have one, and we do not claim to ([CRDT] is cited for the
  general idea of convergent state, not as the mechanism).

## Acceptance criteria

- Kill an expert node: its experts are re-placed within a couple gossip rounds; in-flight
  requests complete; generation output is unchanged (determinism).
- Unit tests for `rebalance_experts` (death, partition, capacity, no-op) and for the
  run-based `ExpertGroup` loading (multiple runs concatenate correctly).
- The routing table swap never exposes a half-applied assignment (test with a
  request storm against a table version bump).

## Risks

- **Churn/flapping.** A noisy load metric could cause constant rebalancing. Mitigation:
  hysteresis (only rebalance past a threshold) and `LeastLoadedPolicy` off by default.
- **Non-contiguous runs vs. `get_slice`.** A node's experts may now be several runs;
  `_load_layer` must concatenate slices and remap slots correctly. Mitigation: covered by
  dedicated loading tests.
