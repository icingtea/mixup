# Open issues & risks in the plan

This is a critical reading of the phased plan. It is deliberately pessimistic: these are
the places where the design is most likely to bite, ordered roughly by how subtle the
failure is. Each item notes the mitigation already in the plan, or what still needs a
decision.

---

## Correctness

### 1. Split-brain is bounded, not prevented

Phase 3 promises "never produce two leaders," but the split-brain section then admits a
symmetric partition elects a leader on *both* sides, and the loser demotes on heal. During
the partition both leaders generate and both write their own ledgers under different
epochs. Invariant #1 is therefore really "at most one leader *per epoch*," not "at most
one leader." The consequence is wasted work (the losing partition's tokens are discarded)
and, worse, the loser may hold the *fresher* ledger at heal time.

*Resolved:* wording qualified to "at most one leader per epoch" in
[architecture.md](architecture.md) (Roles, Invariants, Consistency model) and
[phase-3-leader-election.md](phase-3-leader-election.md) (Goal, "Split-brain: bounded, not
prevented").

### 2. Deterministic election/rebalance needs identical views, but gossip is only eventually consistent

`elect(view)` and `rebalance_experts(view, prev, policy)` are pure functions of `ClusterView`,
and Phase 6 argues "same view → same table." But no two nodes ever hold the *same* view at
the same instant: a node learns a peer is `DEAD` on its own gossip schedule, so node A
rebalances to epoch 5 while node B still holds epoch 4 for a few rounds. That is not a bug
in the function — it is a property of the input — but it means two nodes can compute
different tables (or elect different leaders) with *no* malfunction. The `is_newer_than` /
epoch tie-break reconciles it eventually, but the routing table is **not** a CRDT: it is a
deterministic function of a non-identical input. The plan leans on "convergent/CRDT-style"
without actually having a CRDT.

**Concrete manifestation:** the expert request must carry the routing-table epoch it was
issued under, otherwise a follower that has already swapped to a newer table will apply the
wrong `mine`/`slot` mask and silently produce a wrong partial sum (or double-count). Phase 6
says dispatch "pins the table version," but the wire change (a version field on the request,
and a follower rule to serve-per-epoch or redirect) is never specified.

*Resolved:* [phase-6-dynamic-dispatch.md](phase-6-dynamic-dispatch.md) "Consistency of
the swap (and the wire epoch)" now specifies `ExpertRequest.routing_epoch` +
serve-per-epoch / reject-with-current-epoch.

### 3. The promotion memory story is internally inconsistent

Phase 4 says a follower's `ExpertGroup` is "loaded once at boot and **retained across role
changes**," but also says "the leader holds **no experts**" and invariant #5 says "a node
holds attention/dense tensors only while it is leader." A follower that is promoted while
still holding its expert runs would hold experts + dense simultaneously — violating the
memory accounting. Either promotion must first *unload/hand off* its experts (an extra
rebalance triggered by the promotion itself), or the invariant is wrong.

*Resolved:* promotion is now shed-first — see
[phase-4-lazy-role-loading.md](phase-4-lazy-role-loading.md) (RoleManager) and the election
step in [phase-3-leader-election.md](phase-3-leader-election.md).

### 4. "Identical output" needs bit-exact reduction, not just deterministic greedy decoding

The chaos harness asserts output "must match exactly" after failover. Greedy
decoding is deterministic, but the expert reduce step sums partials **over containers**:
`y = Σ_j y_j`. Rebalancing changes which experts live on which container, which changes the
grouping/order of the float summation, which changes rounding by ~1 ULP, which can flip an
`argmax` token downstream. A correct implementation can still fail an *exact* oracle.

*Resolved (decision):* tolerate the drift. Failover inherently rebalances experts, so the
reduce grouping shifts and a few tokens may differ — accepted as floating-point reality,
not a defect. The chaos harness asserts *completion correctness* (valid, terminating, no
double-count), not token identity. See "Determinism of the reduction (accepted drift)" in
[phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md) and
[testing.md](testing.md).

---

## Performance

### 5. KV-ledger replication is a full snapshot per token

The leader broadcasts the entire `past_key_values` after *every* token to *every* follower.
The cache grows linearly with sequence length, so per-token gossip is
O(layers × tokens × followers) — for a long generation this can rival the expert compute it
exists to protect. The plan bounds *recompute* on failover (≤ a couple tokens) but says
nothing about bounding the *replication* cost of the full snapshot.

*Resolved:* "Replication cost" note added in
[phase-5-kv-cache-ledger.md](phase-5-kv-cache-ledger.md) (delta / gossip-every-N as the
scaling path).

### 6. Control plane and data plane share one loop and one pool

Gossip heartbeats (asyncio, in the FastAPI loop) and expert fan-out (a `ThreadPoolExecutor`)
compete for the same node's CPU and network. A slow expert matmul or a request storm can
delay heartbeats → false-positive `SUSPECT`/`DEAD` → a needless rebalance/election, which is
exactly the cascade Phase 2 worries about. Phase 7 calls the two planes "logically separate"
but does not isolate them (no dedicated worker pool, no prioritization).

*Resolved:* [phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md) now
specifies a dedicated gossip executor + wall-clock heartbeat deadlines (control plane
isolated from data-path load).

---

## Scope & assumptions

### 7. "Minimal loss" depends on greedy/batch=1 being deterministic

The entire Phase 5 recovery story (replay the tail, reproduce logits exactly) rests on
deterministic greedy decoding. The plan pins batch=1/greedy and defers streaming and
speculative decoding — but those are listed as *future triggers*. When sampling or streaming
lands, "recompute the tail exactly" becomes wrong, and the ledger-replay correctness argument
collapses. The determinism assumption is a load-bearing wall, not a temporary simplification.

*Resolved:* load-bearing-assumption callout added in
[phase-5-kv-cache-ledger.md](phase-5-kv-cache-ledger.md) and
[phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md).

### 8. Bully election with a fixed highest-rank rule is fragile to a flaky top node

Rank is `(kv_seq, priority, node_id)` with `priority` static. If the top-priority node is
flaky — dies and resurrects with a higher `incarnation` repeatedly — it keeps winning
elections it cannot hold, producing election storms. Bully elections are known to be fragile
when the highest-ranked node is unreliable; there is no lease length, no minimum term, no
"preferred but degraded" leader. Jitter only breaks *simultaneous* ties, not this.

*Resolved:* "Flapping & cooldown" added in
[phase-3-leader-election.md](phase-3-leader-election.md) (`MIXUP_ELECTION_COOLDOWN` +
lease on a deposed leader).

### 9. Failure model assumes crashes, not misbehavior

"No Byzantine tolerance" is a stated non-goal, and that is fine — but the model also assumes
"a node never sends incorrect messages." A *buggy* node (wrong `kv_seq`, a corrupted routing
table) is not a crash, and with no authentication/encryption (also deferred) a single
misbehaving node can poison membership or the ledger. The apply functions are partly
defensive (monotonic `kv_seq`, `is_newer_than`), which is good, but "fault tolerant" should
be scoped explicitly to *crash-stop + honest bugs*, not silent corruption.

*Resolved:* scoped to "crash-stop + honest bugs" in
[architecture.md](architecture.md) (Failure model), keeping the defensive apply rules as a
non-Byzantine first line of defense.

### 10. Failover has a bounded generation gap; "dynamic" is placement-only

- On leader failure there is a stall of roughly `detection + election + lazy dense load (~1s)
  + rehydration` during which no tokens are produced. That is the accepted availability
  trade-off, but the user-visible stall grows with model size (the dense load) and should be
  stated, not implied.
- "Dynamically distributed" here means *dynamic expert placement across a fixed roster*.
  There is no node autoscaling (the roster is fixed at boot; idle/standby nodes are mentioned
  in Phase 0 but never designed), no cross-host scheduling. That is a reasonable scope, but
  the term over-promises relative to the mechanism.

*Resolved:* a "Bounds & scope" section added to
[architecture.md](architecture.md) stating the failover stall (incl. the shed/rebalance
step) and that "dynamic" is placement-only, no autoscaling.

---

## Already addressed (for the record)

- **No external consumer / service-discovery story.** The plan gives a fault-tolerant cluster
  but nothing tells a user or upstream service *who* the leader is, and the leader moves after
  failover. This gap is closed by the new
  [phase-9-repl-client.md](phase-9-repl-client.md) (`mixup-repl` + `GET /leader` discovery).
