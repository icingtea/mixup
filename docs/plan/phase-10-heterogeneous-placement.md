# Phase 10 — Resource-aware heterogeneous expert placement

> Status: in progress. Build order: Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → (7 deployment, out of
> scope until behaviour is validated on one machine).
>
> Source heuristics: `docs/heuristics/architecture.md`, `docs/heuristics/midsem-report.pdf`
> (same content), and the CoMoE paper (`docs/heuristics/comoe.pdf`).

## Goal

Make expert placement a continuous, resource-aware optimization instead of a static
boot-time assignment, and extend the system to CPU-only devices, replicas, and (later)
quorum/commit fault tolerance. Expert aggregation stays out of scope; the model is never
modified.

## Locked decisions

1. **Leader may host experts** (report Figure 7: "main + experts"). Leader-owning experts
   are served locally; promotion no longer sheds experts.
2. **Full model on every device.** The local `model.safetensors` is the disk tier; any node
   can load any expert locally, so placement changes and replica promotion need no peer
   weight transfer.
3. **Docker per host over SSH** is the eventual deployment (Phase 7), not the dev loop.
4. **Fault tolerance after placement/offload.** Quorum election and commit-before-emit land
   in Phase 5; until then the existing SWIM + epoch election and KV sync apply.

## Test environment

All behaviour is validated on **this machine** (RTX 5060 8GB, 20 cores, 30 GB RAM) using an
emulated heterogeneous pool:

- GPU nodes: containers sharing the 5060, `set_per_process_memory_fraction` + budget.
- CPU nodes: containers with `CUDA_VISIBLE_DEVICES=""`, `--cpus`, `--memory`.
- Disk: app-level `disk_sim_mbps` throttle (deterministic) + optional `--device-read-bps`.
- Network: `tc netem` delay/jitter/loss; partitions via `docker network disconnect`.
- Profile modes: `configured` (decisions use the profile), `measured` (online EWMA),
  `hybrid` (seeded + bounded update, default).

The five-device testbed is final validation (Phase 7), not the development loop.

## Heuristic → code mapping

| Report heuristic | Implementation |
|---|---|
| `R_i(t)={C,M,B}` resource state | `mixup_core.devices` (DeviceProfile, ResourceState) |
| EWMA smoothing | `mixup_core.resource.ewma` |
| Stability `S_r` | `mixup_core.resource.stability` |
| Heterogeneity `H_d` | `mixup_core.resource.heterogeneity` |
| Latency prediction | `mixup_core.cost` (calibrated + online EWMA) |
| Storage ladders | `mixup_node.memory` (Tier, MemoryManager) |
| Loading cost | `mixup_core.cost` |
| Placement optimization | `mixup_core.placement` (trace-driven local search) |
| `P_offload` | `mixup_node.memory` retention score |
| Next-layer prediction | `mixup_node.predictor` (frequency + LR baseline, 2-layer MLP) |
| `θ_pre` prefetch | `mixup_node.memory` (GPU/CPU variants) |
| Eviction score | `mixup_node.memory` |
| Error correction | replica / blocking-load fallback (no expert substitution) |
| Two timescales | coarse placement loop + fine per-token residency |
| Fault tolerance | replicas, commit quorum (majority-ack token log); SWIM+epoch election |

## Phases

- **Phase 0** — measurement & simulation foundation (`mixup-bench`, `mixup-trace`,
  `mixup-sim`), profiles, traces, baselines. Gate: simulator replays the current cluster
  within ±20%.
- **Phase 1** — heterogeneous runtime: CPU expert path, leader-as-owner, config v2,
  `mixup-lab` emulated pool. Gate: 1 GPU + 2 CPU nodes, correct output, kill/failover.
- **Phase 2** — placement engine v1 (stats, cost model, optimizer, replicas, churn).
  Gate: ≥20% p95 improvement over uniform in simulation; real lab cluster matches reference.
- **Phase 3** — tiered residency, offload, prefetch, eviction. Gate: constrained budgets,
  no OOM, prefetch hit rate measured.
- **Phase 4** — dynamic placement, replica promotion, double failure, rejoin.
- **Phase 5** — commit-before-emit via a **commit quorum** (majority-ack token log); keep the
  SWIM+epoch election (no Raft terms/votes, no log compaction).
- **Phase 6** — next-layer expert predictor: frequency baseline + logistic regression +
  2-layer MLP, offline-trained on traces; go/no-go on the MLP.
- **Phase 7** — deployment to the real testbed (deferred).

## Definition of done (every phase)

Unit + property tests, a simulator scenario suite, an integration test, metrics, a design
note. No phase merges on "works on my laptop".

## Notes on the source heuristics

- The report's `θ_pre` formula as written raises the threshold when GPU memory is free,
  contradicting its own text ("more free memory → more aggressive prefetch"). We implement
  the text-consistent form (more free memory → lower threshold) and test monotonicity.
- CoMoE's next-layer predictor assumes the next layer's input is available; in a
  decoder-only model layer `l+1`'s router input depends on layer `l`'s expert output.
  Predictor accuracy is therefore an empirical question; the baseline is frequency/LR.
- Granite 1B-A400M is ~2.6 GB; all five devices fit it in RAM, so real offload pressure is
  created with configurable budgets. The simulator includes larger synthetic MoE models for
  the scaling story.

## Phase 5 design decision: commit quorum, not Raft election

We keep the **SWIM + epoch election** for *who becomes leader* (it is total-ordered,
tested, and split-brain-free for crash-stop failures). For *when a token is safe to emit*
we add a **commit quorum**: a monotonic committed watermark + majority-ack (3 of 5) before
emission, and resume-from-last-committed on failover. This gives "no committed token lost"
and "no split-brain emission" (a partitioned minority can't reach a majority) without Raft
terms, candidate votes, or log compaction. A full quorum *election* is only warranted if we
also want partition-safe *compute*, which we don't for the fixed-roster lab.

## Predictor status (Phase 6, not started)

The consumers (`eviction_score`, `prefetch_threshold`) and the raw material
(`ActivationStats.counts`/`samples`, `mixup-trace` top-k) exist, but no predictor does:

- no `predictor.py` interface;
- no LR (logistic-regression baseline) or NN (CoMoE 2-layer MLP);
- no hidden-state capture (`mixup-trace` records only top-k indices, not `h_t`);
- no layer-l→l+1 transitions (Markov);
- no training/eval harness (top-1/top-3 vs frequency baseline).

Unblock order: (1) hidden-state capture in `mixup-trace`; (2) frequency + LR baseline;
(3) the MLP; (4) eval that decides whether the MLP earns its cost.

### Predictor training: online, on the leader, weights synced

- **Online learning loop** runs on the leader: predict `p(e^{l+1})` at layer `l` from
  `[h_t, e_l, c_t]`, then use the actual top-k at layer `l+1` as the label for one SGD
  step (24 samples/token, no separate pass). The leader is the only place with the router
  and hidden states.
- **Output formulation**: 32 independent binary logits ("is expert i in the top-8?"),
  per-expert binary cross-entropy, thresholded against `θ_pre` — not CoMoE's single
  softmax, which mismatches top-k selection.
- **Order**: LR first (simple online SGD, stable), NN behind the same `Predictor`
  interface; the MLP is the go/no-go.
- **Cold start**: offline pre-train on captured traces (bootstrap) then online fine-tune;
  fall back to frequency until the predictor has enough samples.
- **Weight sync**: a versioned `PredictorState` (monotonic version + serialized weights)
  broadcast every `kv_sync_every` tokens (default 5) via `/predictor`, so a promoted leader
  resumes prefetching from the last synced weights. Weights are tiny (LR ~136 KB, MLP
  ~280 KB). Same monotonic-guard + resume pattern as the KV ledger.
- **Safety**: prefetch is a throughput optimization only — a miss falls back to a blocking
  load, so correctness never depends on the predictor.

## Findings so far (Phase 0)

- **Placement is only non-trivial under a memory budget.** Without one, the optimizer
  (correctly) puts every expert on the single fastest device, because the 1B model fits in
  its VRAM and single-stream latency is minimized by touching the fewest, fastest nodes.
  With a realistic per-node budget (e.g. 800 MB on the real Granite trace), the
  latency-aware placement beats uniform by ~29% (255 ms → 181 ms p50). The budget is what
  forces the heterogeneous spread — and it is exactly the report's memory constraint.
- `mixup-bench` on the 5060 (dedicated, no contention): `expert_fixed_ms ≈ 0.47 ms`,
  `expert_per_ms ≈ 0.038 ms`. The earlier ~1.6–4.6 ms round-trips were shared-GPU
  contention, not the cost of one expert forward.

## Phase 1 validation (single machine, emulated)

- `mixup-lab` stood up 1 GPU leader + 2 CPU expert nodes; generation is coherent and
  correct (the CPU expert path computes in fp32 and matches the fp32 reference exactly).
- Kill the GPU leader → a CPU node is promoted, keeps its 16 experts (leader-as-owner,
  no shedding), loads the dense model on CPU, and keeps generating correctly
  (slowly, as expected). This validates CPU-as-leader and leader-as-owner end-to-end.

## Phase 2 validation (single machine, emulated)

- The leader records activation stats from the router (skipping the batched prefill) and,
  after `REPLAN_MIN_TOKENS` (50), re-places experts via `plan_routing` with a memory
  budget: the 8 hottest experts move to the GPU leader (capped at 604 MB), the rest stay on
  the two CPU nodes (8 / 12 / 12). Post-replan generation is correct and faster, and the
  leader loads its 8 experts on the fly (leader-as-owner). Churn is amortized over the
  placement horizon so the optimizer will actually move experts.

## Phase 3 (tiered residency) — in progress

- Pure core (`tiers.py` + `residency.py`): storage ladders (disk never an execution tier),
  retention/eviction/prefetch scores, and a byte-budgeted residency manager. Unit-tested.
- Node `TieredExpertStore`: per-(layer, expert) lazy load from the model file, eviction
  under budget, fp32 accumulation (matches grouped_mm numerics exactly). CPU + GPU smoke
  tests pass (output matches the reference).
- **Fixed: batched prefill bug.** The store flattened `top_k_index` and applied one weight
  to every token, so the batched prefill produced wrong expert sums and garbage generation
  after re-planning. Now iterates per (token, position).
- **Fixed: stale routing tables.** `/routing` applied any table to the expert group; a
  stale broadcast (lower epoch) could overwrite a newer placement. Now ignores older
  tables and updates the view via `apply_routing`.
- Validated end-to-end: post-replan generation is correct and *faster* than pre-replan
  (the leader's GPU now serves the hot experts instead of the CPU nodes).

## Phase 4 (dynamic placement + replica failover) — in progress

- **Replicas**: `RoutingEntry.backup` + `replica.py` (`plan_backups`, `promote_backups`,
  `failover_with_backups`). Each expert gets a backup on a different device; on primary
  failure the backup is promoted (no re-sharding), and only experts with a dead/absent
  backup fall back to least-loaded re-placement. Pure + tested (including double failure).
- **Periodic re-plan**: the leader re-plans every `REPLAN_INTERVAL_TOKENS` (200) observed
  tokens, not just once, so placement adapts as activation stats drift. Churn is amortized
  in the objective, so re-planning only moves experts when the latency gain outweighs the
  reload cost.
- **Fixed: leader's own store not updated on failover.** `_on_expert_failure` relied on the
  loopback `/routing` broadcast, but that carries the same epoch the view already holds, so
  the staleness guard ignored it. Now updates the leader's store/dispatcher directly.
- **Fixed: scheduler stripped leader-as-owner experts.** `rebalance/balance_experts`
  filtered `role==EXPERT`, so the leader's experts were redistributed away on failover/rejoin.
  Every alive node is now an eligible owner.
- Validated: killing an expert promotes its backups (8 → 13 experts on the leader) with
  full coverage (32) and correct generation.

## Phase 5 (commit quorum) + Phase 6 (predictor) — done, lab-validated

- **Phase 5**: `CommitLog`/`CommitTracker` (pure, tested) + node integration (append/ack/
  replicate via `/commit`, resume-from-committed truncation). Validated: generation and
  leader failover work end-to-end; no committed token is re-emitted (the resume truncates
  to the majority-ack watermark).
- **Phase 6**: `FrequencyPredictor` + `LRPredictor` + `MLPPredictor` (online binary-logit
  SGD), trained on the leader during generation (the dispatcher predicts layer `l+1` from
  layer `l`'s hidden state + experts, label = actual `l+1` top-k) and synced every
  `kv_sync_every` tokens via `/predictor`. Eval on real generation: **LR top-1 0.39 /
  top-3 0.78, MLP 0.34 / 0.69, frequency 0.18 / 0.59** — LR beats frequency ~2×; the MLP
  does not beat LR (the decoder caveat), so LR is the default.

## Remaining

- **Phase 7 real-testbed deployment** (tooling only; hardware required).

## Completed telemetry wiring (post-12h)

- **Prefetch hint** (Phase 3 finish): the leader emits a next-layer prefetch hint carrying
  each candidate's predicted activation probability; the node applies `θ_pre =
  θ_base / (stability · (1 + θ_mem · free_ratio))` before prefetching, and uses the
  probabilities in `S_evict` (was a `0.0` placeholder). Exposed on `/metrics`
  (`residency.prefetch_threshold`).
- **RTT into the cost model**: the dispatcher measures per-node network RTT (expert
  round-trip minus server compute, EWMA) and `replan` feeds it to `plan_routing` (was
  `rtt_ms=0`).
- **Online cost correction**: the dispatcher records `(n_experts, server_ms)` samples and
  fits `expert_fixed_ms + n · expert_per_ms`; `replan` blends the fit with the static
  profile per `profile_mode` (`hybrid` is now the lab default; `measured`/`configured`
  also supported).
- **`H_d` heterogeneity**: computed from the (corrected) profiles and used as a
  conservative tie-breaker in the greedy placement seed (prefer the less-outlier device
  when cost is tied).

## Known residual (boot-time mutual death)

The primary boot-race symptom — a fast CPU node demoting the GPU leader while it loads its
dense model — is fixed by a longer suspect timeout (`SUSPECT_TIMEOUT` 10s → 30s): the GPU
leader now stays leader. What remains: the boot-time rejoin rebalances trigger *simultaneous*
expert reloads on every node, and a node mid-reload can miss gossip pings, briefly
partitioning itself (a node ends up idle until the next replan). Root fix: make expert
reload non-blocking for the gossip path (yield the GIL / background the load), a larger
change tracked separately.

## Known issues (Phase 4)

- **Expert rejoin is fixed for correctness; a residual performance quirk remains.** The
  revival (higher-incarnation), boot grace period, longer timeouts, and **atomic routing
  adoption** (view + dispatcher updated together under the engine lock) now keep the
  leader's dispatcher consistent with its view, so generation is correct after a rejoin.
  What remains: a node *reloading experts* (a multi-second torch load) can still be
  falsely marked DEAD ~50s after boot, which triggers a needless failover that over-loads
  the leader (exceeding its budget) and leaves the peer holding stale, unused experts
  until the next replan. This is a placement-quality/memory issue, not a correctness one.
  Failover (kill → promote backups) is solid; rejoin produces correct output.
