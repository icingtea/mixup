# Deploying mixup across devices

`mixup-lab` can run the cluster two ways, decided by the config file:

- **Single-host lab** (`lab_config.json`) — every node is a container on this machine's
  Docker daemon, sharing the `cluster` network. No `endpoints` in the config.
- **Cross-host deployment** (`devices.json`) — one container per device, each publishing its
  control (`8000`) and expert (`8001`) ports on its own address, with explicit `endpoints`
  so every node can reach every other node.

The same commands work for both:

```bash
mixup-lab doctor devices.json   # validate + show the run plan (no Docker needed)
mixup-lab render devices.json   # print the equivalent `docker run` command per node
mixup-lab up devices.json       # deploy (idempotent; recreates on config change)
mixup-lab status devices.json   # container status per node
mixup-lab down devices.json     # stop + remove the node containers
```

## Prerequisites per device

| Device | Docker | SSH | GPU |
|---|---|---|---|
| this PC (leader) | nix Docker | — | CDI |
| Windows 11 (3060) | Docker Desktop (WSL2) | OpenSSH Server | `--gpus all` |
| Raspberry Pi (8 GB) | `virtualisation.docker` | sshd | none (CPU) |

On Windows: enable **OpenSSH Server** (Settings → Optional features), make sure `docker.exe`
is on the SSH session's `PATH`, and verify `ssh <user>@<host> docker version` works. Verify
the GPU with `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`.

On the Pi (NixOS):

```nix
services.openssh.enable = true;
virtualisation.docker.enable = true;
users.users.<you>.extraGroups = [ "docker" ];
```

## Image: build multi-arch and push to GHCR

The image must run on both `linux/amd64` (this PC, the laptop) and `linux/arm64` (the Pi).
Build once with buildx and push to GitHub Container Registry:

```bash
gh auth refresh -s write:packages        # once, so we can push packages
echo "$(gh auth token)" | docker login ghcr.io -u icingtea --password-stdin
docker buildx build --platform linux/amd64,linux/arm64 \
  --tag ghcr.io/icingtea/mixup:latest --push .
```

Each device pulls the right architecture automatically. If the package is private, each
device must `docker login ghcr.io` first; simplest is to make the package public.

## SSH access

`mixup-lab` uses the Docker SDK's `ssh://` transport (`docker system dial-stdio` over SSH),
so the daemon is reached the same way on Linux, Windows, and the Pi — no exposed Docker
socket. Put your public key on each device:

```bash
ssh-copy-id <user>@pi-inch-nails
ssh-copy-id <user>@monad-windows-old   # if OpenSSH is running
```

`devices.json` carries the SSH username in `specs.<node>.docker_host`; edit it to match your
accounts. Docker-over-SSH requires the `paramiko` extra, which is already a dependency of
`mixup-orchestrator`.

## Config

`devices.json` is the working example (tailnet IPs from `tailscale status`). The parts you
will most likely edit:

- `specs.<node>.docker_host` — SSH user / device name.
- `specs.<node>.gpu_mode` — `cdi` (Linux), `nvidia` (Docker Desktop), `none` (CPU).
- `profiles.<node>` — seed numbers; `profile_mode: hybrid` corrects them online from
  measured dispatch latencies, so they do not need to be exact.
- `startup.experts` — the initial expert partition (contiguous, covering `0..31`). The
  leader re-places from activation stats at runtime, so this is only the boot layout.

The model is downloaded once per device into a named volume (`mixup-lab up` runs a one-shot
`mixup-node-fetch` container when the volume is empty). The leader's local node bind-mounts
the existing `models/` directory instead.

## Web console

Point the web server at the deployment config:

```bash
cd apps/mixup-web && MIXUP_CONFIG=../../devices.json npm run dev
```

The console discovers every node via the `endpoints` map, and its stop/start buttons drive
the right daemon through `docker -H`.

## Validating failover

```bash
mixup-repl --config devices.json "hello"
mixup-lab down devices.json        # or use the web stop button for a single node
# ... watch the leader re-elect and rebalance, then:
mixup-lab up devices.json
```

## Notes

- **Security**: the control and expert ports have no authentication. On the tailnet, ACLs
  are the perimeter; expose 9000/9001 only to trusted peers.
- **EC2**: install Tailscale on the instance so it sits on the same private `100.x` network
  and reuse the same URL scheme (the config is not Tailscale-specific). A home PC behind NAT
  cannot accept inbound connections from a public EC2 host without an overlay or port-forward,
  which is why Tailscale/WireGuard is the simple answer.
- **Latency**: dispatch is per-layer, so WAN distance matters. Home tailnet is fine; a
  far-away EC2 region will be slow (placement will still prefer hot experts on the leader).
