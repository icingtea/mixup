# Phase 2 — Gossip membership & failure detection (SWIM)

**Goal:** replace the static boot roster with live, decentralized membership: every node
tracks which peers are alive, and the cluster converges on a shared view even under
crashes and message loss — without a coordinator.

## Why SWIM

SWIM — *Scalable Weakly-consistent Infection-style Process Group Membership* — is the
de-facto standard for failure detection in peer-to-peer clusters ([SWIM]). It is what
HashiCorp's Serf/Consul memberlist implement ([Serf-gossip], [memberlist]). Chosen
because:

- It needs **no coordinator**: detection is peer-to-peer, matching decision #4.
- It **scales**: failure detection cost is O(1) per node per round (not O(n) broadcast
  heartbeats), irrelevant at 5 nodes but right by construction.
- It is **tunable** in its false-positive rate, and separates *detection* (who might be
  down) from *dissemination* (gossip), which we need to treat independently.

## Failure detection (SWIM)

SWIM distinguishes a crashed node from a slow or partitioned one by using **indirect
probing**:

1. **Direct ping** — node `A` sends `Ping` to `B` and waits `PING_TIMEOUT`. If `B`
   replies `Ack`, `B` is alive.
2. **Indirect probe** — if no `Ack`, `A` picks `k` random peers and sends them
   `PingReq(target=B)`. Each peer pings `B` and, if it gets an `Ack`, sends `A` an `Ack`
   on `B`'s behalf. If **any** indirect `Ack` arrives, `B` is still alive (the direct
   path was just flaky).
3. **Suspect** — only if *no* ack (direct or indirect) arrives does `A` mark `B`
   `SUSPECT` and gossip `Suspect(B)`.
4. **Dead** — a suspect is declared `DEAD` only after `SUSPECT_TIMEOUT`, during which
   `B` can clear the suspicion by sending an `Alive` (e.g. after a GC pause or a
   partition heals).

This is the standard defense against false positives. We add the **incarnation counter**
so a restarted node (same `node_id`, higher `incarnation`) is recognized as fresh and can
revive itself from `DEAD`.

## Gossip dissemination (anti-entropy)

Membership changes (and later, routing-table versions) spread by **push-pull gossip**:
each node periodically picks a random peer and exchanges summaries, then pulls what it's
missing. Push-pull converges in O(log n) rounds with high probability ([Demers],
[Birman]).

In our implementation the gossip payload carries, alongside the SWIM status, the sender's
`kv_seq` and `routing_epoch`, so **membership, ledger freshness, and routing version ride
the same message** — a single gossip round serves all three concerns (see Phases 3, 5, 6).

## Transport

HTTP, consistent with the rest of the stack (see the transport decision in
[phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md)). Each node exposes:

```
GET  /health                     # liveness for Docker/ops
POST /gossip                     # {Ping|Ack|PingReq|Suspect|Alive|Dead} → reply
POST /routing                    # push a newer RoutingTable
GET  /routing?epoch=N            # pull table if sender is ahead (or full pull)
```

The `mixup-node` `gossip.py` transport layer is a thin wrapper: it takes the pure
`apply_gossip` from `mixup-core` and feeds it real network events on a background task
(asyncio, in the same event loop as FastAPI). This separation is the whole point of
Phase 1 — the state machine is unit-tested in core, the transport is a dumb pipe.

## Tunables (new env vars, documented in `.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `MIXUP_GOSSIP_INTERVAL` | 1.0 s | anti-entropy round period |
| `MIXUP_PING_TIMEOUT` | 0.5 s | direct probe timeout |
| `MIXUP_INDIRECT_K` | 2 | peers asked in an indirect probe |
| `MIXUP_SUSPECT_TIMEOUT` | 3.0 s | suspect → dead grace period |
| `MIXUP_DEAD_TIMEOUT` | 5.0 s | prune dead members after this long |

## Methodology notes

- **Eventually-perfect failure detector.** SWIM's suspicion phase is what makes it a
  *◇P*-style detector in the [Chandra-Toueg] taxonomy: eventually, every crashed node is
  permanently suspected by some correct node, and eventually, no correct node is
  suspected. We document the *bounds* (time to detection) as `PING_TIMEOUT +
  k·PING_TIMEOUT + SUSPECT_TIMEOUT` in the worst case.
- **Phi-accrual as a possible refinement.** A fixed `PING_TIMEOUT` trades false
  positives against detection latency. The φ-accrual failure detector ([PhiAccrual]) —
  used by Akka and Cassandra — adapts the threshold to the observed inter-heartbeat
  distribution, reducing false positives under load. We do **not** adopt it in v1
  (fixed timeouts are simpler to reason about and test), but the `Ping/Ack` timing data
  is recorded so φ-accrual can be dropped in later without a protocol change.
- **No gossip encryption/auth** in v1 — the network is a private Docker bridge; a
  signature/HMAC option is a documented future hardening step, not in scope.

## Acceptance criteria

- Kill any expert node: within `~3 × gossip interval` all live nodes agree it is `DEAD`,
  with no false `DEAD` when a node merely pauses (simulated).
- The pure `apply_gossip` passes a table of state-machine transition tests.
- Two nodes that disagree on membership converge after a few gossip rounds (simulated in
  the deterministic test harness).

## Risks

- **False positives cascade.** If a slow node is marked `SUSPECT` and then `DEAD` too
  fast, we'd trigger a needless rebalance/election. Mitigation: the suspicion grace
  period + incarnation revival, and conservative defaults.
- **Reordering vs. incarnation.** An old `Dead` message must not override a newer
  `Alive`. Mitigation: every status message carries `incarnation`, and transitions only
  move forward in `(incarnation, status)` order.
