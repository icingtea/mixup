# Phase 5 — KV-cache ledger: replication & resume

**Goal:** survive leader failure with *minimal* loss of already-computed tokens, by
replicating the KV cache as a monotonic ledger and rehydrating it on the elected
successor.

## The problem

Autoregressive generation is a function of the **KV cache** (`past_key_values`): the
key/value activations for every attention head at every layer, accumulated over all
preceding tokens. Recomputing the cache from scratch after a leader failure means
re-running the entire prefix forward pass — including the expert fan-out — which is
exactly the work we want to avoid. For this 1B model the KV cache is small (≈5 MB at
~100 tokens), so replicating it is cheap and is the right recovery primitive.

## Design: a replicated, leader-authoritative log

The leader is the **single writer** of an append-only, sequence-numbered log (the
"ledger"). This is the Raft *log replication* shape — one leader appends, followers
replicate — but here the log entries are KV-cache snapshots, and replication is
best-effort (see the failure model in [architecture.md](architecture.md)):

```
kv_seq:    1        2        3       ...      n
        ┌───────┬───────┬───────┬─────┬───────────┐
        │ cache │ cache │ cache │ ... │ cache     │   (append-only)
        └───────┴───────┴───────┴─────┴───────────┘
                          │
              leader writes; followers hold the freshest entry
```

- **Write path.** After generating token `t`, the leader serializes `past_key_values` to
  a safetensors blob (keys `layers.{i}.k` / `layers.{i}.v`), builds the `LedgerHeader`
  (`kv_seq = t`, `input_ids`, `position_ids`, `cache_position`, `attention_mask`,
  `layer_shapes`), and gossips `(header, blob)` to peers.
- **Replication.** Followers store the latest `(header, blob)` they've seen — they do
  **not** need attention layers to hold the bytes, which is why a follower stays minimal
  (Phase 4). Each follower advertises its `kv_seq` in gossip so its election rank is
  always current.
- **Monotonicity.** `kv_seq` never decreases; a follower discards any entry `≤` its
  current `kv_seq` (invariant #3).

The "consensus" requested is therefore: **gossip gives consensus on *who the leader is***
(Phase 3), and the leader gives a *single authoritative ordering* of ledger entries —
two different concerns, two different mechanisms, one clean separation.

## How well this replicates (fidelity)

The KV cache is the **only cross-token state**: embeddings, router, and the experts are all
stateless and recomputed each step. Preserving the cache therefore preserves *all* the
prefix progress — the entire prefix's worth of attention work is kept, and only the
un-replicated tail (see "Bounding lost work") is recomputed.

Three properties make the replicated cache trustworthy even though it is **not**
bit-identical to the dead leader's:

1. **Single writer.** Only the leader appends, so there is one authoritative ordering;
   followers never merge conflicting caches.
2. **Snapshot, not deltas.** Each entry is a whole, self-consistent cache, so there is no
   torn or partial state — a follower holds a complete `(header, blob)` pair or none.
3. **Valid, not identical.** Rehydration must produce a cache that is *self-consistent*
   with its own header (same `kv_seq`, shapes, `position_ids`), not one that matches the
   dead leader's bytes. Every node reads the same read-only weights, so the cache is
   consistent with the weights the successor loads and the continuation is valid. The
   header's `layer_shapes` and the pinned transformers version guard the only real risk —
   a format/shape mismatch — not value fidelity.

## Resume (rehydration)

On promotion, the new leader (already elected for highest `kv_seq`, Phase 3):

1. Sheds its own experts and lazily loads the dense model (Phase 4) — fresh attention
   layers with empty caches.
2. Fetches the freshest ledger blob it holds (or pulls it from a peer, see below).
3. **Rehydrates** `past_key_values`: for each layer, sets the attention module's
   `past_key_value` to the replicated `(k, v)` tensors, on the right device/dtype.
4. Resumes `generate` from the last `input_ids` with `past_key_values` supplied,
   `position_ids`/`cache_position` from the header, and `max_new_tokens` reduced by the
   tokens already produced.

Rehydration uses transformers' standard `past_key_values` API — the cache format is
documented and stable enough for this ([HF-generation], [HF-llm-optimization]).

## Bounding lost work

The only progress lost on failover is the **tail** — the tokens the dead leader generated
after its last successful gossip. Those are re-run by the successor; the *amount* of
recompute is bounded by the **gossip lag** — at most a couple of tokens with per-token
gossip, configurable (`MIXUP_KV_GOSSIP_EVERY`):

| `MIXUP_KV_GOSSIP_EVERY` | Recompute on failover |
|---|---|
| 1 (default) | ≤ 1–2 tokens |
| N | ≤ N tokens |

**Replication cost.** The same table reads the *recompute* bound, but the *replication*
cost is a full `past_key_values` snapshot per gossip round — O(layers × tokens ×
followers), growing linearly with sequence length. For a 1B model at short lengths this is
negligible (≈5 MB at ~100 tokens), but the scaling path is a **delta** (gossip only the
newest token's K/V entries) or a larger `MIXUP_KV_GOSSIP_EVERY`. Documented, not in v1
scope.

The header's `input_ids` makes the resume point auditable and lets a lagging successor
pull a missing blob from a peer with a higher `kv_seq` (the pull path reuses
`GET /routing`-style gossip but for the ledger).

## Two recovery paths (both are "minimal loss", different trade-offs)

This is the "do both" decision, decomposed:

1. **Replicated KV cache** — the primary path: rehydrate the freshest blob, resume with
   ≈zero recompute. This is what makes failover fast.
2. **Token-sequence replay** — the fallback/audit path: if no blob is available (e.g.
   all replicas lag), replay `input_ids` from the header through a fresh dense forward to
   rebuild the cache, then continue. Zero tokens are *lost* (they're all in `input_ids`);
   only recompute time is paid.

Path 1 is the normal case; path 2 is the correctness backstop. Its audit — replay must
reproduce a fresh forward pass under the *same* placement — validates the replay mechanism,
not bit-identity with the pre-crash run (which may have used a different placement).

> **Determinism note.** Recovery produces a *valid* continuation; it does not promise bit-
> or token-identity with the pre-crash run (failover rebalances experts and shifts the
> reduce grouping — see Phase 7). Deterministic greedy decoding (batch=1, `do_sample=False`)
> is still assumed so a repeated run under an *unchanged* placement is reproducible and
> auditable. When streaming or sampling lands, that audit no longer applies and must be
> revisited (see [phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md)).

## Ledger storage

- The blob lives in the **read-write ledger volume** (bind-mounted, Phase 8) so a
  restarted node can recover its last ledger without asking a peer, and so the leader can
  cheaply re-read its own last entry.
- In-memory, the ledger is a single-entry ring (only the freshest entry matters) plus a
  short history of `input_ids` for audit.

## Methodology notes

- The **"freshest log wins"** election rule is Raft's §5.4.1 election restriction applied
  to a KV-cache log ([Raft]). The analogy is exact: a node that hasn't seen the latest
  entries shouldn't become leader, because it would resume from stale state.
- **Append-only + monotonic seq** is the standard replicated-log design ([Raft],
  [Lamport] total order). We need only *one* entry at a time (the latest), so we drop
  Raft's full log-compaction machinery.
- **safetensors** for the blob (not pickle) for the same untrusted-input safety as the
  existing wire format ([safetensors]).

## Acceptance criteria

- Steady state: after each token, followers hold the leader's `kv_seq` (eventually).
- Kill the leader: the elected successor rehydrates the freshest `kv_seq` and **continues
  generation correctly** — output is a valid, terminating completion, and the prefix is
  preserved (only the bounded tail is recomputed). Output is not required to be
  token-identical to a no-failure run (see Phase 7).
- Replay path (no blob available) reconstructs a valid cache and continues; audited against
  a fresh forward pass under the same placement, not against the pre-crash run.

## Risks

- **`past_key_values` format drift** across transformers versions. Mitigation: pin
  transformers (already pinned in `uv.lock`), and gate on the version in the header.
- **Rehydration device/dtype mismatch.** Mitigation: the header records shapes; the node
  casts/transfers to the model's dtype/device on load.
- **Best-effort replication can lose the tail.** Accepted by design; bounded to the gossip
  lag (see above).
