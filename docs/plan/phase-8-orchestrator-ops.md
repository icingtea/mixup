# Phase 8 — Orchestrator & ops

**Goal:** adapt the host-side `mixup-orchestrator` to the new runtime: read
`startup_config.json`, start uniform nodes, provide the ledger volume, pass through the
GPU, and stop treating the leader as a one-shot process.

## Changes to `distribute.py`

1. **Read `startup_config.json`** (`load_startup_config`, Phase 0). The `DEFAULT_CONFIG_PATH`
   and the `--config` arg update accordingly.
2. **Uniform node image.** Every container runs the same image with the same
   `MIXUP_NODE_ENV`-style payload, now carrying `StartupConfig` + `node_id` + `incarnation`
   (+ `priority`). No per-role images; roles are decided at runtime (Phase 3/4).
3. **GPU passthrough.** Add `device_requests`/`--gpus all` so the CUDA wheel in the image
   can actually use the GPU. This is the earlier open TODO; it requires the NVIDIA
   Container Toolkit on the host ([nvidia-toolkit], [docker-gpu]). On a CPU-only host the
   containers still build/start; they just won't find CUDA.
4. **Ledger volume.** Add a second bind mount (read-write) for the KV-cache ledger,
   distinct from the read-only `models/` mount. Path e.g. `./var/ledger` → `/usr/local/app/ledger`.
   This is where Phase 5 writes its blob.
5. **Restart policy.** Keep `restart_policy: unless-stopped` — now that the node is a
   persistent server (not a one-shot generator), it won't loop. **Resolves ADR-011.** The
   *leader-election* logic (not Docker restart) is what handles a node actually dying; a
   container restart is just a fast local recovery of a follower (re-joins gossip with a
   new `incarnation`).
6. **Network.** Reuse the `cluster` bridge network as today. Gossip runs over HTTP on the
   bridge, using container-name hostnames — no IP bookkeeping.

## The node entrypoint (`mixup-node`)

`infer.py` stops branching once at boot (`central` vs `expert`) and instead:

1. Boots as a **follower** (loads experts only), starts the FastAPI server (health, gossip,
   leader, expert, generate endpoints). `GET /leader` is a read-only projection of the
   node's `ClusterView` (`{leader, epoch, kv_seq, routing_epoch}`) used by external clients
   (Phase 9) to discover the leader without joining gossip.
2. Starts the gossip task (Phase 2) and election watcher (Phase 3).
3. Promotes/demotes under the `RoleManager` (Phase 4) as the `ClusterView` dictates.

The leader's generation is a **background task** on the same event loop; `POST /generate`
on the leader queues/triggers it. The `PROMPT`/`MAX_NEW_TOKENS` env passthrough (ADR-014)
is retained and now applies to the leader's generate loop.

## `.env.example` additions

All new tunables are documented in one place (the ones introduced across Phases 2–7):
`MIXUP_GOSSIP_INTERVAL`, `MIXUP_PING_TIMEOUT`, `MIXUP_INDIRECT_K`,
`MIXUP_SUSPECT_TIMEOUT`, `MIXUP_DEAD_TIMEOUT`, `MIXUP_ELECTION_TIMEOUT`,
`MIXUP_ELECTION_JITTER`, `MIXUP_KV_GOSSIP_EVERY`, `MIXUP_EXPERT_TIMEOUT`,
`MIXUP_MAX_RETRIES`, `MIXUP_HTTP_POOL_SIZE`, `MIXUP_HTTP2`.

## Ops runbook additions (`docs/running.md`)

- **Failover drill (expert):** `docker stop container_2` → watch the scheduler re-place
  experts 0–7 and generation continue.
- **Failover drill (leader):** `docker stop container_1` → watch an election, lazy dense
  load, ledger rehydration, and resume.
- **Chaos script:** a small `scripts/chaos.sh` (see [testing.md](testing.md)) to
  kill/partition nodes on a timer for the integration harness.

## Methodology notes

- **GPU in Docker** requires `nvidia-container-toolkit` + `--gpus all`; reference the
  NVIDIA container toolkit install guide and Docker's GPU resource docs ([nvidia-toolkit],
  [docker-gpu]).
- **Bind-mounted shared state** for the ledger follows the same pattern as the read-only
  model mount; both rely on the single host being the source of truth for weights/state
  (no multi-host storage in scope).

## Acceptance criteria

- `mixup-orchestrator` starts a uniform cluster from `startup_config.json`; `docker logs
  container_1` shows the leader elected and generating.
- `docker stop` of any node triggers the corresponding failover path without manual steps.
- GPU is visible in a leader container (`nvidia-smi`) when the toolkit is present.

## Risks

- **GPU passthrough is host-dependent.** Documented as conditional; a CPU-only fallback
  (or CPU torch) is a separate, optional path, not part of this phase.
- **Two mounts (models ro + ledger rw)** must not be confused; the ledger path is
  writable and owned by the `app` user in the image.
