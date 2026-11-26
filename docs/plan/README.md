# mixup — dynamic experts & failover: implementation plan

This directory is the detailed implementation plan for the next major iteration of
`mixup`: **dynamic expert loading, peer-to-peer membership, freshness-aware leader
election, and crash recovery with minimal loss of computed tokens**.

It is a *plan*, not yet code. It sits alongside the shipped docs in `docs/` (which
describe the system as it exists today) and supersedes them where they conflict.

## The four decisions that shape this work

These were settled explicitly during planning; everything below follows from them:

1. **Minimal memory footprint.** A node loads only what it needs from the shared
   read-only `models/` mount. Expert nodes load only their expert slices. A node loads
   the attention/dense layers **only when it becomes (or is becoming) the leader**, and
   unloads them when it demotes.
2. **KV cache as a replicated ledger.** The elected leader is the authoritative
   writer; it broadcasts the latest KV cache after each token. Followers hold the bytes
   as a replicated, monotonic ledger even though they don't load attention layers. The
   freshest ledger wins elections.
3. **Complete dynamic expert dispatch.** Expert placement is owned by a pluggable
   scheduler driven by heuristics (load, latency, capacity, memory). Failover is just
   the first heuristic.
4. **Peer-to-peer gossip.** Membership, failure detection, and routing-table
   propagation are decentralized. There is no static coordinator; the elected leader is
   only the *generation* authority.

A fifth, transport-level decision is documented in
[phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md): we stay on
HTTP/FastAPI (with connection pooling + deadlines) rather than moving to gRPC.

## Goals

- Survive the crash of any **expert** node: its experts are re-placed onto survivors
  and generation continues without restarting the cluster.
- Survive the crash of the **leader**: a new leader is elected, lazily loads the
  attention/dense layers, rehydrates the freshest replicated KV cache, and resumes
  generation with **minimal** (bounded, deterministic) recompute.
- Keep per-node **memory footprint** at the minimum for its current role.
- Make expert placement **a policy**, not a hardcoded partition, so heuristics can be
  swapped and tuned.
- Keep all the *logic* (membership, election, scheduling, ledger format) in the
  **torch-free** `mixup-core` so it is unit-testable without a GPU or Docker.

## Non-goals (for this iteration)

- **No Byzantine fault tolerance.** Nodes fail by crashing or partitioning; they don't
  lie. We assume crash-stop semantics (see the failure model in
  [architecture.md](architecture.md)).
- **No multi-datacenter / WAN.** Everything runs on one Docker bridge network.
- **No cross-host GPU scheduling.** Placement is about *experts*, not about scheduling
  GPU jobs across machines.
- **No speculative/batched decoding** yet. Generation stays greedy, batch=1 for now;
  streaming and speculative decoding are noted as future triggers (see
  [phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md)).
- **No weight synchronization.** Expert weights come from the shared read-only mount;
  the only replicated runtime state is the KV-cache ledger.

## Glossary

| Term | Meaning |
|---|---|
| **Node** | One `mixup-node` process in a Docker container. |
| **Leader** | The elected node that owns generation and is the authoritative KV-cache writer. |
| **Follower** | Any non-leader node; serves experts and replicates the KV ledger. |
| **Incarnation** | A node's boot counter, incremented each restart, to distinguish a restarted node from a stale one. |
| **Epoch** | A monotonic logical clock of leadership generations, incremented each election. |
| **`kv_seq`** | The monotonic sequence number of the KV-cache ledger. |
| **RoutingTable** | The versioned `ExpertId → (NodeId, local_slot)` map, propagated by gossip. |
| **Run** | A contiguous range of expert ids owned by one node (a node may own several runs). |
| **StartupConfig** | The initial, static layout from `startup_config.json`. |
| **SWIM** | Scalable Weakly-consistent Infection-style membership protocol (failure detection). |
| **Ledger** | The append-only, sequence-numbered KV-cache log replicated by the leader. |

## Phase roadmap

| # | Phase | Depends on | Deliverable |
|---|---|---|---|
| 0 | [Rename `Config` → `StartupConfig`](phase-0-startup-config-rename.md) | — | Pure refactor, no behavior change |
| 1 | [Runtime state model](phase-1-runtime-state-model.md) | 0 | Core types + pure functions (torch-free) |
| 2 | [Gossip membership (SWIM)](phase-2-gossip-membership.md) | 1 | Failure detection + anti-entropy |
| 3 | [Freshness-aware leader election](phase-3-leader-election.md) | 2 | Election with split-brain protection |
| 4 | [Lazy role loading](phase-4-lazy-role-loading.md) | 1, 3 | On-demand attention/dense load & unload |
| 5 | [KV-cache ledger replication & resume](phase-5-kv-cache-ledger.md) | 3, 4 | Replicated ledger + rehydration |
| 6 | [Dynamic expert dispatch](phase-6-dynamic-dispatch.md) | 2, 4 | Scheduler/placement policy + run-based loading |
| 7 | [Dispatch resilience & transport](phase-7-dispatch-resilience.md) | 6 | Retries, deadlines, connection pooling |
| 8 | [Orchestrator & ops](phase-8-orchestrator-ops.md) | 0 | `startup_config.json`, GPU passthrough, ledger volume, `GET /leader` |
| 9 | [REPL client (`mixup-repl`)](phase-9-repl-client.md) | 8 | Host-side CLI: leader discovery + `POST /generate` |
| 10 | [Testing & documentation](testing.md) | all | Torch-free unit tests, chaos harness, doc updates |

**Ordering note.** Phases 4–7 are partly parallelizable, but the value lands through a
thin vertical slice: get *rename → membership → election → lazy load → ledger resume*
working end-to-end on the existing single-expert-per-node layout *before* investing in
the scheduler heuristics (Phase 6). Phase 0 should be landed as its own small commit —
it touches many files but changes no behavior.

## Reading order

1. [architecture.md](architecture.md) — the target system and its invariants.
2. Phases 0–10, in order, for the detailed design of each.
3. [testing.md](testing.md) — how each piece is verified.
4. [issues.md](issues.md) — a critical reading: open problems and risks.
5. [references.md](references.md) — the consolidated bibliography.
