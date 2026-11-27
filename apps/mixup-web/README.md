# mixup-web

Web GUI for the mixup cluster — a SvelteKit (Svelte 5 + Chart.js) replacement for the
Textual `mixup-tui`. One origin, so no CORS: the SvelteKit server aggregates the node
HTTP APIs, measures control-plane RTT, drives `docker stop/start`, and proxies
`/generate` with failover.

## Run

Requires Node 20+ and the lab running (`nix develop --command mixup-lab`).

```bash
cd apps/mixup-web
npm install        # first time only
npm run dev        # http://127.0.0.1:5173
```

Production build:

```bash
npm run build
node build         # serves on PORT (default 3000)
```

## Configuration

Node discovery reads `lab_config.json` (or `startup_config.json`), found by walking up
from the working directory — so running from the repo root or `apps/mixup-web` both
work. Override with `MIXUP_CONFIG=/abs/path/to/config.json`.

## Tabs

- **Chat** — send a prompt to the leader (failover-aware), like the REPL.
- **Nodes** — per-node role / leader / execution tier / expert count / cpu / gpu / rtt / tokens.
- **Resources** — CPU, memory, GPU-free and RTT time series per node (the "resource
  perception" view).
- **Experts** — top experts by activation (bar chart + table).
- **Predictor** — live predictor kind/version and top-1 / top-3 accuracy.
- **Summary** — cluster, predictor and phase summary.
- **Local** — the "no distribution" baseline: runs the whole model single-process on a
  GPU node (inside its container) for a side-by-side tok/s comparison.

The sidebar shows every node's liveness, the leader badge, and kill/start toggles that
map to `docker stop` / `docker start`.

## Server API

| Route | Method | Purpose |
| --- | --- | --- |
| `/api/cluster` | GET | full snapshot (nodes, leader, predictor, experts, routing, history) |
| `/api/generate` | POST | `{prompt, max_new_tokens}` → generation result |
| `/api/nodes/:name/:action` | POST | `action` in `stop` \| `start` |
| `/api/local` | POST | `{prompt, max_new_tokens}` → no-distribution baseline |
