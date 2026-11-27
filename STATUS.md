# Status — handoff for the next agent

> Written by the previous agent at commit `40e2353`. Everything below is committed;
> the working tree is clean. Read `docs/plan/phase-10-heterogeneous-placement.md` for the
> full phased plan (it is the living design doc, and the repo owner has been updating it).

## TL;DR — what I was doing right now

I was implementing **Phase 5 (commit-before-emit via a commit quorum)**. The pure core is
done and tested; the node integration is **committed but NOT yet lab-validated** — my
validation run was interrupted mid-build.

- Last commit: `40e2353 feat(node): commit quorum integration (Phase 5)`.
- Immediately before that: `acdd640` (CommitTracker) and `677c92e` (CommitLog/quorum).

## Project

`mixup`: distributed inference of a MoE LLM (IBM Granite 3.1 1B-A400M: 24 layers, 32
experts, top-8) across heterogeneous devices, with resource-aware expert placement,
offloading, and fault tolerance. Goal: fault-tolerant inference across heterogeneous
devices (a 5060 laptop + 3070 + 3060 + 2 Raspberry Pi 8GB — but **all development/testing
happens on this one machine** via emulated devices; real deployment is Phase 7).

## Branch state

- Branch: `feat/heterogeneous-placement`, **53 commits ahead of `master`**, clean tree.
- **255 tests pass, ruff clean.** (The test venv is `/tmp/mixup-test-venv`, torch-free.)
- Note: the repo owner made concurrent commits (`b7cdff1`, `35d222d`) updating the plan
  doc and starting the `CommitTracker` wiring. My commits build on top of those.

## Architecture

5 uv workspace packages (the `mixup-gateway` package was removed — its failover logic
moved into the REPL):

| Package | Role |
|---|---|
| `mixup-core` | pure pydantic/logic (torch-free): types, runtime, gossip, election, scheduler, placement, cost, residency, replica, tokenlog, sim, planner |
| `mixup-node` | torch runtime: `infer.py` (FastAPI app), `node.py` (Node), `dispatch.py`, `model.py`, `memory.py`, `monitor.py`, `gossip_loop.py` |
| `mixup-orchestrator` | host-side: `distribute.py` (Docker), `lab.py` (`mixup-lab` emulated pool) |
| `mixup-repl` | client CLI (`ClusterClient` with failover retry) |

Key new modules (this branch):

- `mixup-core/devices.py` — DeviceProfile/ResourceState/ExecutionType/ProfileMode
- `mixup-core/resource.py` — EWMA/stability/heterogeneity
- `mixup-core/stats.py` — ActivationStats + trace loader
- `mixup-core/cost.py` — latency model
- `mixup-core/placement.py` — Uniform/Frequency/Latency policies + p95 optimizer
- `mixup-core/sim.py` — deterministic simulator
- `mixup-core/tiers.py` + `residency.py` — storage tiers + residency decisions
- `mixup-core/replica.py` — backup assignment + promotion + failover
- `mixup-core/tokenlog.py` — CommitLog + CommitTracker (Phase 5)
- `mixup-node/memory.py` — TieredExpertStore (GPU/CPU/disk residency)
- `mixup-node/monitor.py` — live CPU/mem/GPU sampling

## What's done (Phases 0–4 + start of 5)

- **Phase 0** — measurement/simulation: device profiles, resource math, activation stats,
  cost model, placement optimizer, simulator, `mixup-bench`/`mixup-trace` CLIs, real trace
  fixture.
- **Phase 1** — heterogeneous runtime: CPU expert path (fp32 compute, bf16 wire),
  leader-as-owner, `profiles`/`profile_mode`, `LabConfig`, `mixup-lab`.
- **Phase 2** — smart placement: leader records router top-k, re-plans from stats+profiles+
  memory budget (hot experts → GPU leader). Churn amortized over the placement horizon.
- **Phase 3** — tiered residency/offload: `TieredExpertStore` (lazy load/evict, fp32
  accumulation matching grouped_mm), retention/eviction/prefetch scores.
- **Phase 4** — replica-aware fault tolerance: `RoutingEntry.backup`, backup promotion,
  atomic routing adoption (view+dispatcher under the engine lock), periodic re-plan,
  gossip boot grace period + longer timeouts.
- **Phase 5 (in progress)** — commit quorum: `CommitLog`/`CommitTracker` (pure, tested) +
  node integration (committed, unvalidated).

### Bugs fixed this session (important context)
- Enum `is` vs `==` comparison.
- Churn not amortized (optimizer never moved experts).
- Tiered store flattened batched prefill → garbage output.
- Stale routing tables overwriting newer ones.
- Leader's own store not updated on failover (loopback broadcast ignored as stale).
- `balance/rebalance_experts` stripped leader-as-owner experts.
- Leader promotion didn't `promote()` (view role stayed EXPERT) → REPL got 503s.

## Phase 5 design (per the owner's plan commits)

Keep the **SWIM + epoch election** for *who becomes leader*; add a **commit quorum** for
*when a token is safe to emit*: a monotonic committed watermark + majority-ack (3 of 5)
before emission, resume-from-last-committed on failover. No Raft terms/votes/log compaction.

What I implemented:
- `mixup-core/tokenlog.py`: `quorum_size`, `CommitRecord`, `CommitLog` (append/advance/
  committed_prefix/truncate_to), `CommitTracker` (append + `record_ack` + `record_acks_through`
  + majority-ack watermark).
- `mixup-node/node.py`: `self._commit_log = CommitLog()` (all nodes), `self._commit =
  CommitTracker(n, log=self._commit_log)` (leader, in `_become_leader`); `_generate` appends
  each token + self-acks; `_replicate_commit(past, nxt, input_ids, max_new_tokens)` sends
  records + aligned KV snapshot to followers and advances the watermark on majority-ack;
  `store_commit(body)` (follower) stores records + KV snapshot, returns the acked index.
- `mixup-node/infer.py`: `POST /commit` endpoint.

**Not yet done / not validated:** the resume path does NOT yet truncate to the committed
watermark (the KV snapshot is now aligned at commit time, but the warm-resume logic in
`_generate` still resumes from `self._kv` without consulting `self._commit_log.committed`).
The commit-quorum integration has **not been run in the lab** (the build/test was cut off).

## Known issues (from `docs/plan/phase-10-heterogeneous-placement.md`)

- **Expert rejoin correctness is fixed** (higher-incarnation revival + atomic adoption);
  a residual *performance* quirk remains: a node reloading experts can still be falsely
  marked DEAD (~50s after boot, not during the 15s grace window), triggering a needless
  failover that over-loads the leader beyond its budget. Not a correctness bug.

## How to run / test

```bash
# unit tests (torch-free venv)
PYTHONPATH="packages/mixup-core/src:packages/mixup-node/src:packages/mixup-repl/src:packages/mixup-orchestrator/src" \
  /tmp/mixup-test-venv/bin/python -m pytest tests -q

# lint
ruff check .

# everything below needs the Nix dev shell (bare python/uv fail on NixOS):
nix develop --command <cmd>

# build the node image (torch/CUDA, ~4 min first build, fast after)
docker build --network=host --tag mixup:latest .

# start the emulated lab (1 GPU leader container_1 + 2 CPU experts container_2/3;
# config in lab_config.json; host ports 9000/9001/9002)
nix develop --command mixup-lab

# prompt it (REPL auto-discovers leader and retries across failover)
nix develop --command mixup-repl --config lab_config.json "What is the capital of France?"

# inspect
curl localhost:9000/metrics   # live resource perception + profile + stats
curl localhost:9000/routing   # expert -> {node, slot, backup}
curl localhost:9000/store     # leader's own expert store
curl localhost:9000/leader
curl -X POST localhost:9000/replan   # force placement re-plan
```

Docker logs: `docker logs container_1` (leader) etc. Metrics ledger: `var/ledger/metrics_*.jsonl`.

## Next steps (suggested order)

1. **Validate the Phase 5 commit quorum** in the lab: rebuild the image, run `mixup-lab`,
   generate, and confirm (a) the `/commit` round-trip works, (b) the committed watermark
   reaches the token count, (c) kill the leader mid-generation and confirm resume honors
   the committed watermark.
2. **Wire the resume truncation**: in `_generate`'s warm-resume branch, consult
   `self._commit_log.committed` and truncate `generated_ids` (and align the KV snapshot) so
   resume is exactly the committed prefix.
3. **Phase 6 (predictor)** — the owner's plan commits (`b7cdff1`, `35d222d`) spell out the
   direction: online LR-first predictor on the leader (predict `p(e^{l+1})` from `[h_t, e_l, c_t]`,
   label = actual top-8 at l+1, 32 independent binary logits), versioned `PredictorState`
   synced every `kv_sync_every` tokens, cold-start from offline pre-train + frequency
   fallback. Unblock order: (1) hidden-state capture in `mixup-trace` (currently only top-k),
   (2) frequency + LR baseline, (3) MLP, (4) eval.
4. Phase 7 — real testbed deployment (deferred).

## Gotchas

- **The repo owner may commit concurrently** (they have been, to the plan doc + node.py).
  Re-check `git log`/`git pull` before starting.
- The node/torch code only runs inside Docker; the test venv can't import `mixup_node`'s
  torch modules. Torch-dependent checks go in `scripts/*.py`, run inside a container
  (e.g. `docker run --rm --device nvidia.com/gpu=all -v "$PWD/packages:/usr/local/app/packages:ro"
  -v "$PWD/models:/usr/local/app/models:ro" -v "$PWD/scripts/X.py:/tmp/s.py:ro" --entrypoint
  python mixup:latest /tmp/s.py`). `mixup-bench`/`mixup-trace` run the same way.
- `startup_config.json` is the classic flat config; `lab_config.json` is the emulated-lab
  config (`{startup, profiles, specs, profile_mode}`). `load_startup_config()` accepts both.
- Failover is slow by design (~15s detection from 5s/10s gossip timeouts + ~5s dense load);
  don't treat that latency as a bug.
