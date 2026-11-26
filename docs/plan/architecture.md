# Target architecture

This is the architecture the phased plan builds toward. It is a superset of today's
system (see `docs/architecture.md` for what exists now); everything new is marked.

## Roles

```
                    gossip: membership, loads, kv_seq, routing table
        ┌─────────────────────────────────────────────────────────────────┐
        │        (peer-to-peer; no static coordinator)                    │
        ▼                                                                 ▼
 ┌─────────────────┐                                      ┌─────────────────┐
 │  follower node  │◄────────── gossip (SWIM) ───────────►│  follower node  │
 │  experts only   │                                      │  experts only   │
 │  + KV ledger    │                                      │  + KV ledger    │
 └────────┬────────┘                                      └────────┬────────┘
          │      ┌──────────────────────────────────┐              │
          └─────►│  LEADER (elected, role not id)   │◄─────────────┘
                 │  loads attention+dense lazily    │
                 │  drives generation               │
                 │  authoritative KV-cache writer   │
                 │  dispatches expert requests      │
                 └──────────────────────────────────┘
```

- **Follower (expert) node** — the default, minimal-footprint state. Loads *only* its
  expert slices from the shared read-only mount. Serves `POST /{layer}`. Replicates the
  KV-cache ledger as opaque bytes (a few MB) and reports `kv_seq` in gossip. Does **not**
  load embeddings, attention, router, or norms.
- **Leader** — the elected generation authority. On promotion it first sheds its own
  experts to survivors (a rebalance), then lazily loads the dense + attention + router +
  norms + LM head, then drives autoregressive generation,
  dispatching expert work to followers. After each token it writes and broadcasts the
  updated KV-cache ledger. On demotion it unloads the dense layers to return to minimal
  footprint.

There is **at most one leader per epoch** (enforced by epochs), and every node is
eligible to be elected — no node is special at the identity level. In the rare case of a
symmetric network partition, both sides may briefly elect a leader (one per epoch); the
loser demotes on heal (Phase 3).

## What lives in which package

The split rule from the existing design is preserved and sharpened: **everything that
can be pure logic lives torch-free in `mixup-core`**; anything touching tensors, torch,
transformers, or sockets lives in `mixup-node`; host-only Docker/HF work lives in
`mixup-orchestrator`.

| Concern | Package | New? |
|---|---|---|
| `StartupConfig`, `NodeId`, `Epoch`, `NodePriority`, `NodeRole` | core | rename/extend |
| `MemberState`, `RoutingTable`, `ClusterView` | core | new |
| Gossip state machine (SWIM) | core | new |
| Election logic (freshness + priority) | core | new |
| Scheduler / `PlacementPolicy` + `rebalance_experts` | core | new |
| KV-ledger header codec (schema, not tensors) | core | new |
| HTTP gossip/heartbeat/routing transport | node | new |
| Lazy role load/unload (`roles.py`) | node | new |
| KV-ledger tensor replication + rehydration | node | new |
| Resilient `Dispatcher` | node | rewrite |
| FastAPI server (health, gossip, leader, expert, generate) | node | extend |
| Role-aware entrypoint (persistent server + generation task) | node | rewrite |
| Orchestrator (`startup_config`, GPU, ledger volume) | orchestrator | extend |
| REPL client (`mixup-repl`): leader discovery + `POST /generate` | repl | new |

## Invariants

These must hold at all times; they are the acceptance criteria for correctness.

1. **At most one leader per epoch.** A node acts as leader only if its current epoch
   matches the highest epoch it has observed and it won that election.
2. **Every expert is served.** At any time the union of all nodes' owned experts equals
   `0 .. num_local_experts-1` (coverage), with no duplicate owner (uniqueness).
3. **The ledger never regresses.** `kv_seq` is monotonic; a node discards any ledger
   entry with a sequence number ≤ the one it already holds.
4. **No un-owned experts during a rebalance.** The routing table swap is atomic and
   versioned; a request is always answered by exactly one owner under one table version.
5. **Minimal footprint.** A node holds attention/dense tensors only while it is leader.

## Failure model

We assume the standard **crash-stop + partition** model, the same model SWIM and Raft
are designed for ([SWIM], [Raft]):

- A node either works correctly or stops sending messages (crash). It never sends
  incorrect messages (no Byzantine behavior).

Fault tolerance is therefore scoped to **crash-stop + honest bugs** (a node may crash,
stall, or hit an exception — but it is assumed not to silently corrupt state). The
monotonic `kv_seq` and `is_newer_than` guards in the apply functions are a cheap first
line of defense against a buggy node, not a Byzantine-tolerance mechanism.
- The network may drop, delay, or reorder messages, and may partition the cluster.
- Messages are not corrupted (transport checksums cover this).

This lets us use *unreliable failure detection* ([Chandra-Toueg]): we can only suspect
a node is down, never know for certain, so the protocol must be safe even if a
suspicion is wrong (see [phase-2-gossip-membership.md](phase-2-gossip-membership.md)).

## Consistency model

- **Membership & routing** are **eventually consistent** and propagated by gossip.
  There is no strong agreement; each node acts on the latest version it has seen.
  Convergence is driven by anti-entropy (push-pull gossip) ([Demers]).
- **Leadership** is **eventually single** via epochs: stale-epoch leaders yield, so
  after a bounded time all live nodes agree on one leader (see
  [phase-3-leader-election.md](phase-3-leader-election.md)).
- **The KV ledger** has a **single writer** (the leader) and is replicated to
  followers. Replication is best-effort (the tail may lag), which is exactly why
  election prefers the freshest ledger — the "freshest-wins" rule mirrors Raft's
  election restriction ([Raft] §5.4.1).

## Bounds & scope

- **Failover is placement-only, not autoscaling.** "Dynamic" means expert *placement*
  across a fixed boot roster changes at runtime; the roster itself is fixed (no node
  autoscaling, no cross-host scheduling).
- **Leader failover has a visible stall.** On leader loss, no tokens are produced for
  roughly `detection + election + shed/rebalance + lazy dense load (~1 s) + rehydration`.
  This is the accepted availability trade-off (CAP/PACELC) and grows with model size (the
  dense load dominates).

## Lifecycle

1. **Boot** — every container reads `StartupConfig` + its own `node_id` + `incarnation`
   from the injected environment, loads only its experts, and starts a persistent
   FastAPI server (no more one-shot central; resolves the old ADR-011).
2. **Join & gossip** — nodes exchange SWIM membership and the routing table.
3. **Elect** — the initial leader is `StartupConfig.initial_leader`; thereafter
   elections are freshness-aware. The leader lazily loads dense layers.
4. **Generate** — the leader runs `POST /generate` (or an internal loop) and dispatches
   expert work; after each token it appends to and broadcasts the KV ledger.
5. **Expert failure** — SWIM declares a follower dead; the scheduler re-places its
   experts; the new routing table gossips out; dispatch retries against the new owner.
6. **Leader failure** — SWIM declares the leader dead; the freshest follower wins the
   election, sheds its experts (a rebalance), lazily loads dense layers, rehydrates the
   ledger, and resumes.
7. **Demote** — a deposed leader unloads dense layers and becomes a follower.

## End-to-end data flow (one token, steady state)

For token `t` at layer `l` (same as today's flow, but with a live routing table):

1. Attention + norm run on the leader.
2. Router picks top-k experts and their weights.
3. The leader consults the **live `RoutingTable`** and fans the request out to the
   owners (in parallel, over a persistent connection pool).
4. Each owner runs only its own experts and returns a partial sum.
5. The leader sums partials, produces logits, and samples the next token (greedy).
6. The leader appends `(kv_seq+1, past_key_values, input_ids, …)` to the ledger and
   gossips it.

See [phase-5-kv-cache-ledger.md](phase-5-kv-cache-ledger.md) for the failover flow.

## What changes vs. today (cheat sheet)

| Today | Becomes |
|---|---|
| static `Config` | `StartupConfig` (initial) + live `RoutingTable` |
| `build_lut` once at boot | versioned `RoutingTable`, gossiped |
| `ExpertGroup` loads one contiguous slice | loads a set of **runs**, swapped under lock |
| `Dispatcher` holds `self.lut`/`self.containers` | reads live table; retries + deadlines + pooling |
| one-shot central (ADR-011) | persistent leader server with `POST /generate` |
| no liveness/election/checkpoint | SWIM + freshness election + KV ledger |
| no external client | `mixup-repl` CLI that discovers the leader and prompts it |
