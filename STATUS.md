# Status — handoff for the next agent

> Everything below is committed; the working tree is clean. Read
> `docs/plan/phase-10-heterogeneous-placement.md` for the full phased plan.

## TL;DR — project state after the 12-hour sprint

The heuristics plan is **~90% done**. Phases 0–6 are implemented and lab-validated; what
remains is Phase 7 (real-testbed deployment, needs hardware) plus two minor Phase 3
telemetry wirings.

- Branch: `feat/heterogeneous-placement`, clean tree.
- **259 tests pass, ruff clean.** (Test venv `/tmp/mixup-test-venv`, torch-free.)

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

## Remaining

1. **Phase 3 telemetry wiring** (minor): feed measured RTT into the cost model (currently
   `rtt_ms=0`), online EWMA correction of `expert_fixed_ms`/`expert_per_ms`, use the
   heterogeneity score `H_d` in placement.
2. **Phase 7 deployment** (needs the physical 3070/3060/2×Pi testbed): multi-host
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
```

Docker logs: `docker logs container_1` (leader) etc. Metrics ledger: `var/ledger/metrics_*.jsonl`.

## Gotchas

- Torch code only runs inside Docker; the test venv can't import `mixup_node`'s torch
  modules. Torch checks go in `scripts/*.py`, run via `docker run --rm --device
  nvidia.com/gpu=all -v "$PWD/packages:/usr/local/app/packages:ro" -v
  "$PWD/models:/usr/local/app/models:ro" -v "$PWD/scripts/X.py:/tmp/s.py:ro" --entrypoint
  python mixup:latest /tmp/s.py`.
- `lab_config.json` is the emulated-lab config (`{startup, profiles, specs, profile_mode}`);
  `load_startup_config()` accepts both it and the flat `startup_config.json`.
- Failover is slow by design (~15s detection from 5s/10s gossip timeouts + ~5s dense load).
- Known residual quirk: a node reloading experts can be falsely marked DEAD ~50s after boot
  (not in the 15s grace window), causing a needless failover that over-loads the leader.
  Performance issue, not correctness.
