# Running

## Prerequisites

- A NixOS host with Docker running (`docker info`).
- The model files on the host (downloaded once).

## 1. Fetch the model

```bash
nix develop
mixup-fetch-model            # downloads to models/granite-3.1-1b-a400m-instruct/
```

This runs `huggingface_hub.snapshot_download` against
`ibm-granite/granite-3.1-1b-a400m-instruct`. Override with `--model` / `--dir`.

> The node reads a **single consolidated** `model.safetensors` (see ADR-003). If the HF
> repo is sharded, the model needs consolidating first.

## 2. Verify the routing (smoke test)

```bash
mixup-smoketest
```

Prints the expert-to-node assignment and verifies coverage/contiguity (exit 0 = OK):

```
central: container_1
experts:
  container_2: [0..7]    (8 experts)
  container_3: [8..15]   (8 experts)
  container_4: [16..23]  (8 experts)
  container_5: [24..31]  (8 experts)
OK: 32/32 experts covered, no overlaps, contiguous per node
```

## 3. Start the cluster

```bash
mixup-orchestrator
```

This (in order): checks the model exists, builds the image (streaming the Docker build
log live), creates/reuses the `cluster` network, and starts `container_1` (central) +
`container_2..5` (experts), mounting `models/` read-only.

The **first** build is slow — it downloads the GPU torch wheel (~540 MB) + the nvidia
packages. Subsequent builds are cached.

## 4. Watch the forward pass

```bash
docker logs -f container_1    # central: "container_N is up", then the generated text
docker logs container_2       # expert: "container_2: experts 0-7 ready on port 8000"
```

The central waits for all experts to come up, then generates one completion and prints
it. Coherent output = the distributed forward pass works.

## 5. Stop / clean up

```bash
docker stop container_1 container_2 container_3 container_4 container_5
docker rm   container_1 container_2 container_3 container_4 container_5
```

## Fast forward-pass test

Generation is 50 tokens × 24 layers of HTTP round-trips, so it's slow. Do a 1-token
smoke instead:

```bash
mixup-orchestrator --max-new-tokens 1
```

(Changing `--prompt`/`--max-new-tokens` requires recreating the containers — stop/rm
first, or run on a fresh cluster.)

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `mixup-orchestrator` → "model not found" | run `mixup-fetch-model` first |
| `nix develop` fails on `nvidia-cufile` / `libmlx5.so.1` | don't build the node in Nix (ADR-006); the dev shell already excludes it |
| `nix develop` fails on `ModuleNotFoundError: editables` | build backend must be `uv_build`, not hatchling (ADR-010) |
| Central log shows the generation repeating forever | the central is one-shot + `restart_policy: unless-stopped` (ADR-011, open) — stop `container_1` or remove its restart policy |
| Experts "up" but central produces nothing | generation is just slow; check `docker stats` (experts at high CPU = working), or use `--max-new-tokens 1` |
| "Local directory not empty" from fetch | `models/…` already has files; clear it or use a fresh `--dir` |

## Architecture summary

See [architecture.md](architecture.md) for the full data flow and lifecycle.
