# Phase 3 — Freshness-aware leader election

**Goal:** when the leader dies (or is partitioned away), the surviving nodes elect
exactly one new leader per epoch, preferring the node whose replicated KV-cache ledger is
most up-to-date. Split-brain is *bounded* (at most one leader per epoch), not made
impossible — see below.

## What "leader" means here

The leader is the **generation authority**: the only node that runs the dense/attention
forward pass, dispatches expert requests, and writes the KV-cache ledger. Membership and
failure detection remain peer-to-peer (Phase 2); leadership is a *role* layered on top,
not a coordinator role for membership.

## Why freshness-aware

The election goal is not just "pick a live node" but "pick the node that can resume
generation with the least recompute." The KV ledger is a monotonic, sequence-numbered log
(`kv_seq`), so the node holding the highest `kv_seq` has seen the most up-to-date state.
Preferring it minimizes lost work on failover — this is decision #2 and it directly
parallelizes Raft's rule that a candidate with the most up-to-date log wins
([Raft] §5.4.1).

## Election algorithm

A **bully-style** election (highest-ranked live node wins, [Garcia-Molina]) with a
**freshness-then-priority** ranking:

```
rank(m) = (m.kv_seq, m.priority, m.node_id)     # lexicographic, higher wins
```

1. **Trigger.** A node observes the leader is `DEAD` (via `apply_gossip`) or that the
   leader's `epoch` is stale.
2. **Nominate.** The observer computes `elect(view)` locally — the highest-ranked `ALIVE`
   member. If that's itself, it proposes; otherwise it waits (a higher-ranked node will
   propose). Nodes that think they can win enter `CANDIDATE`.
3. **Claim.** A candidate bumps `epoch = current_epoch + 1` and gossips
   `LeaderClaim(node_id, epoch, kv_seq)`.
4. **Acknowledge / yield.** A node that sees a `LeaderClaim` with a higher rank or a
   higher epoch yields. A node that sees a claim *lower* than its own rank responds with
   its own claim (higher rank wins), suppressing the lower one.
5. **Win.** A candidate that receives no higher-ranked claim within
   `ELECTION_TIMEOUT` declares itself leader, sets `role = LEADER`, sheds its own experts
   to survivors (Phase 4, a rebalance), lazily loads the dense layers, then announces
   `LeaderWin(epoch)`.
6. **Resume.** The winner rehydrates the freshest ledger (Phase 5) and continues
   generation.

The **lexicographic rank** over `(kv_seq, priority, node_id)` is a total order, so
exactly one node is the maximum — the election is deterministic even if messages are
delayed. Methodology references: [Garcia-Molina] (bully), [Raft] (log-up-to-date
restriction + randomized timeout to break ties).

## Split-brain: bounded, not prevented

Split-brain (two leaders at once) is *bounded*, not made impossible, by **epochs** and
the **rank order** — there is no quorum:

- Every leadership claim and every routing table carries an `epoch`. A node only acts as
  leader while its epoch is the highest it has observed; on seeing a higher epoch it
  demotes and unloads dense layers (Phase 4).
- A resurrected old leader has a stale epoch and immediately yields on its first gossip
  round, so it never generates concurrently with the new leader.
- To avoid two nodes in different partitions both declaring themselves leader, the winner
  requires **no contradiction** within `ELECTION_TIMEOUT`; in the rare case of a
  symmetric partition where both sides elect someone, the **higher epoch wins on
  healing**, and the loser demotes. The same epoch-tie is broken by rank, so convergence
  is guaranteed (cf. [Raft]'s randomized election timeout for the equivalent liveness
  argument).

We accept a brief window of **unavailability** (no leader) rather than risk two leaders —
availability is sacrificed for safety, the standard CAP/PACELC trade-off for a
single-writer system ([Brewer-CAP], [PACELC]).

## Flapping & cooldown (anti-storm)

Bully elections are fragile when the highest-ranked node is *flaky*: a node that dies and
resurrects (higher `incarnation`) repeatedly keeps winning elections it cannot hold. Two
rules blunt this:

1. **Cooldown on failure.** A node that was `DEAD` (or just lost leadership) within
   `MIXUP_ELECTION_COOLDOWN` is ineligible to win — its claims are ignored until the
   cooldown expires. A brief *lease* makes a deposed leader ineligible to re-win
   immediately.
2. **Jitter still breaks simultaneous ties** (unchanged); the cooldown breaks the
   repeated-flaky-winner loop, which jitter cannot.

| Var | Default | Meaning |
|---|---|---|
| `MIXUP_ELECTION_COOLDOWN` | 10 s | a recently-failed node's ineligibility window |

## Interaction with the KV ledger

- A node's `kv_seq` is gossiped (Phase 2), so every node's `rank` is known globally
  without extra messages.
- The winner may be *behind* the dead leader by the un-replicated tail (the tokens the
  dead leader computed but hadn't gossiped yet). The ledger header carries `input_ids`,
  so the winner regenerates that bounded tail deterministically (greedy decoding is
  deterministic). "Minimal loss" = loss bounded by the gossip lag, recomputed exactly.
  See [phase-5-kv-cache-ledger.md](phase-5-kv-cache-ledger.md).

## Election state machine

```
FOLLOWER ── leader DEAD / stale epoch ──► CANDIDATE ── no higher claim in timeout ──► LEADER
    ▲                                        │                                         │
    └──────── higher epoch / rank seen ◄─────┴─────────────────────────────────────────┘
```

The transition function is pure (in `mixup-core.election`) and table-tested like the SWIM
machine.

## Tunables

| Var | Default | Meaning |
|---|---|---|
| `MIXUP_ELECTION_TIMEOUT` | 1.5 s | max wait for a higher-rank claim |
| `MIXUP_ELECTION_JITTER` | ±20% | randomized backoff to break symmetric ties |

## Acceptance criteria

- Kill the leader: a single new leader is elected within a few seconds, and the other
  survivors agree on it (at most one leader per epoch), verified under message loss in
  simulation.
- The winner is the highest-`kv_seq` survivor (assert with differing `kv_seq` seeds).
- A resurrected old leader demotes itself on seeing a higher epoch.
- The pure `elect`/transition functions pass table-driven tests.

## Risks

- **Symmetric partitions** (both sides elect). Mitigated by epoch + rank; the window is
  bounded and self-heals. We document this explicitly rather than claim it's impossible.
- **Election storms** after recovery. Mitigated by rank determinism + jitter.
