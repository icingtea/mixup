# Consensus over the network + warm KV-cache sharing

## Problem
Two things currently rely on a shared Docker bind-mounted volume (works on one host,
breaks when nodes live on different devices):

1. The KV-cache/checkpoint (`var/ledger` → `/usr/local/app/ledger`) — the leader writes
   `checkpoint.json` locally; a promoted leader reads it from the shared volume.
2. The SWIM gossip consensus is fully implemented + tested but **never wired into the
   node** — membership/failure detection is a 1s `/health` poll, not the SWIM protocol.

## Goals
- Run the SWIM gossip loop on every node so membership + failure detection + `kv_seq`
  (election freshness) propagate over the network.
- Broadcast the leader's KV cache to followers every N tokens (configurable), so a
  promoted leader resumes **warm** (no replay) without a shared volume.

## Stage 1 — wire up consensus (gossip loop)
- Add a thread-safe `Cluster` holder (view + `GossipEngine` + lock) so the node and the
  gossip loop share one view. Refactor `Node` to read/mutate the view through it.
- Start a `GossipLoop` on every node; give it a `send` that POSTs `/gossip` to peers and
  an `on_dead` callback for failure reaction.
- Expose `/gossip` on the node's FastAPI app (merge the `gossip_http` handler).
- Reaction on DEAD: leader dead → elect + promote; expert dead → rebalance.
- Propagate `kv_seq`: the leader bumps its own `MemberState.kv_seq` as it generates; the
  gossip `Ping`/`Ack` already carry `kv_seq`, so the freshest node wins the election.
- Replace the 1s `/health` watchdog with SWIM probe-based failure detection.

## Stage 2 — warm KV-cache sharing
- Replace `model.generate` with an explicit generation loop (like `.temp/`'s) so the
  leader holds the `DynamicCache` and can snapshot it every N tokens.
- Serialize the cache (`[(layer.keys, layer.values)]` → safetensors, spiked OK) and
  broadcast it to followers over the network; followers keep the freshest entry.
- On promotion, the new leader loads the freshest cache via
  `DynamicCache(ddp_cache_data=...)` and continues generating from `kv_seq` — no replay.
- Add a config knob (e.g. `kv_sync_every` tokens) to `StartupConfig`/`NodeEnv`.
- Remove the shared-volume checkpoint as the resume path (keep as a local fallback).

## Risks / decisions
- The explicit generation loop must reproduce `generate`'s greedy output exactly; I'll
  verify it bit-for-bit against `generate` before landing it.
- The `Cluster` refactor touches the Node's view mutations; keep the existing failover
  tests green and re-run the manual drills after.
- KV-cache broadcast is ~50 KB/token (bf16, 24 layers); every 5 tokens ≈ 250 KB/frame —
  fine for LAN and adequate for WAN at this model size.
