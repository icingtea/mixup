# Status — handoff for the next agent

> Everything below is committed; the working tree is clean. Read
> `docs/plan/phase-10-heterogeneous-placement.md` for the full phased plan.

## TL;DR — project state

The heuristics plan is **~90% done**. Phases 0–6 are implemented and lab-validated, merged
to `master`, and code-cleaned. What remains is Phase 7 (real-testbed deployment, needs
hardware) plus two residual quirks (boot-settling, distributed BF16 non-determinism).

- Branch: `master` (feature branch `feat/heterogeneous-placement` merged).
- **238 tests pass, ruff clean.** (Test venv `/tmp/mixup-test-venv`, torch-free.)

## Surface area (post-cleanup)

The codebase is reduced to the runtime + the three user-facing entry points:

- `mixup-core` — pure pydantic models (strict validation via `mixup_core.base.StrictModel`),
  cost model, placement, residency, gossip/election, commit log.
- `mixup-node` — in-container runtime (torch/CUDA): `mixup-node`.
- `mixup-orchestrator` — host-side lab: `mixup-lab`, `mixup-fetch-model`.
- `mixup-repl` — CLI benchmark client (failover-aware): `mixup-repl`.
- `mixup-web` — SvelteKit web UI (chat, metrics, local baseline).

Removed (dead / superseded / eval-only): `mixup-tui`, `mixup-place`, `mixup-smoketest`,
`mixup-bench`, `mixup-trace`, `mixup-orchestrator`, the simulator (`sim.py`), trace
generation (`tracegen.py`), device-spec presets (`profiles.py`), the old HTTP gossip
transport (`gossip_http.py`), and `scripts/*` (smoke tests + offline predictor training).

## Done (all lab-validated on this machine)

- **Phase 0** — device profiles, resource math, activation stats, cost model, placement
  optimizer, simulator, `mixup-bench`/`mixup-trace`.
- **Phase 1** — CPU expert path (fp32 compute, bf16 wire), leader-as-owner, `mixup-lab`.
- **Phase 2** — activation-stats re-planning (hot experts → GPU leader), churn amortization.
- **Phase 3** — `TieredExpertStore` (GPU/CPU/disk residency), retention/eviction/prefetch
  scores, **prefetch hint wiring** (leader predicts next-layer experts, node prefetches).
- **Phase 4** — replica-aware failover (backups + promotion), atomic routing adoption,
  periodic re-plan, gossip grace period + longer timeouts.
- **Phase 5** — commit quorum: `CommitLog`/`CommitTracker` (pure) + node integration
  (append/ack/replicate via `/commit`, resume-from-committed truncation). No Raft
  terms/votes (kept SWIM + epoch election for *who* leads; quorum for *when* to emit).
- **Phase 6** — predictor: `FrequencyPredictor` + `LRPredictor` + `MLPPredictor` (online
  binary-logit SGD), trained on the leader during generation, weights synced every
  `kv_sync_every` tokens via `/predictor`, prefetch hint piggybacked on expert requests.
  Eval on real generation: **LR top-1 0.39 / top-3 0.78, MLP 0.34 / 0.69, frequency
  0.18 / 0.59** — LR wins; MLP does not beat LR (decoder caveat), so LR is the default.
- **UI** — `mixup-web` (SvelteKit + Chart.js) replaces the Textual `mixup-tui`: chat +
  node liveness/kill-start sidebar, per-node CPU/mem/GPU/RTT time series, expert
  activation charts, predictor accuracy, summary, and a "no distribution" local-baseline
  tab. The SvelteKit server aggregates the node APIs (one origin, no CORS) and proxies
  `/generate` with failover.

## Remaining

1. **Boot-settling quirk** — the first generation after a fresh `mixup-lab` can be
   short/odd while the cluster rebalances; subsequent generations are clean.
2. **Distributed BF16 non-determinism** — the same prompt can give different (each coherent)
   outputs across runs because the distributed reduce rounds differently after a re-plan.
3. **Phase 7 deployment** (needs the physical 3070/3060/2×Pi testbed): multi-host
   Docker-over-SSH, CUDA/ARM image variants, model sync, real-device calibration, rerun the
   validation matrix. Tooling can be written here; final validation needs the hardware.

## How to run / test

```bash
# unit tests (torch-free venv)
PYTHONPATH="packages/mixup-core/src:packages/mixup-node/src:packages/mixup-repl/src:packages/mixup-orchestrator/src" \
  /tmp/mixup-test-venv/bin/python -m pytest tests -q
ruff check .

# everything below needs the Nix dev shell:
nix develop --command <cmd>

docker build --network=host --tag mixup:latest .   # node image (torch/CUDA)
nix develop --command mixup-lab                    # emulated lab (9000/9001/9002)
nix develop --command mixup-repl --config lab_config.json "prompt"  # client w/ failover retry

curl localhost:9000/metrics   # live resource perception + profile + stats
curl localhost:9000/routing   # expert -> {node, slot, backup}
curl localhost:9000/store     # leader's own expert store
curl -X POST localhost:9000/replan

# web GUI (SvelteKit)
cd apps/mixup-web && npm install && npm run dev   # http://127.0.0.1:5173
```

Docker logs: `docker logs container_1` (leader) etc. Metrics ledger: `var/ledger/metrics_*.jsonl`.

## Gotchas

- Torch code only runs inside Docker; the test venv can't import `mixup_node`'s torch
  modules (the test venv is torch-free).
- `lab_config.json` is the emulated-lab config (`{startup, profiles, specs, profile_mode}`);
  `load_startup_config()` accepts both it and the flat `startup_config.json`.
- Failover is intentionally conservative (~20s: 5s ping + 5s indirect probe + 10s suspect
  timeout, then the dense-model load). The clients (REPL/web) retry with backoff for long
  enough to absorb it; a leader killed mid-generation resumes from the warm KV snapshot.
- Core models are strict (`mixup_core.base.StrictModel`): no implicit type coercion, except
  `StrEnum` fields (declared `Field(strict=False)`) because JSON carries enum values as
  strings, and `RoutingTable.assignment` keys (coerced from string by a validator).
- Known residual quirk: a node reloading experts can be falsely marked DEAD ~50s after boot
  (not in the 15s grace window), causing a needless failover that over-loads the leader.
  Performance issue, not correctness.
