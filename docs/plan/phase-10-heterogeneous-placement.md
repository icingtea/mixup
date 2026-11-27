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
| Next-layer prediction | `mixup_node.predictor` (frequency/Markov baseline, MLP optional) |
| `θ_pre` prefetch | `mixup_node.memory` (GPU/CPU variants) |
| Eviction score | `mixup_node.memory` |
| Error correction | replica / blocking-load fallback (no expert substitution) |
| Two timescales | coarse placement loop + fine per-token residency |
| Fault tolerance | replicas, quorum, token commit log |

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
- **Phase 5** — quorum election + commit-before-emit.
- **Phase 6** — predictor MLP (go/no-go).
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
  Predictor accuracy is therefore an empirical question; the baseline is frequency/Markov.
- Granite 1B-A400M is ~2.6 GB; all five devices fit it in RAM, so real offload pressure is
  created with configurable budgets. The simulator includes larger synthetic MoE models for
  the scaling story.

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

## Known issues (Phase 4)

- **Expert rejoin (restarting a killed node) is flaky.** A restarted node's revival
  sometimes doesn't stick — the gossip loop re-marks it dead before its own gossip loop
  starts (which begins only after model load). Failover works; rejoin needs the gossip
  loop to start before model load, or an explicit revival retry. The fresh time-based
  incarnation is in place (so a restart is distinguishable) but the timing still races.
