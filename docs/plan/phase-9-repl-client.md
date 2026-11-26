# Phase 9 — REPL client (`mixup-repl`)

**Goal:** a small, torch-free host-side CLI that lets a human talk to the running cluster:
discover the current leader, send an arbitrary prompt, and print the answer — transparently
surviving leader failover.

## Why

The plan produces a fault-tolerant cluster but no *consumer* story. Generation is triggered
by `POST /generate` on the leader, but nothing tells a user (or an upstream service) *who*
the leader is — and after failover it is a different container. The REPL is both the human
interface and the first external client: it proves the cluster is usable from outside even
as the leader moves.

It is deliberately **not** part of `mixup-node`. `mixup-node` is the server (it serves
experts, gossips, elects). The REPL is a client: it joins nothing, holds no weights, and its
only jobs are *find the leader* and *ask it a prompt*.

## What it is / is not

- **A client.** No torch, no model weights, no serving. Pure HTTP + pydantic; depends on
  `httpx` (connection pooling + deadlines, cf. [phase-7-dispatch-resilience.md]) and
  `mixup-core` (for the wire contract and address helpers).
- **Not on `mixup-node`.** It does not join gossip, hold experts, or run in a model
  container.
- **v1 is single-shot Q&A** — prompt in, completion out — consistent with the plan's
  non-goals (greedy, batch=1, no streaming). Multi-turn conversation is maintained
  *client-side* by re-sending history with each prompt; reusing the leader's KV cache across
  turns is a future optimization (see open questions).

## Leader discovery

Each node exposes a tiny read-only endpoint that projects its `ClusterView` (Phase 1):

```
GET /leader   → { leader: NodeId | None, epoch: Epoch, kv_seq: int, routing_epoch: int }
```

This is a *read* of state the node already maintains; calling it requires no gossip
membership. The endpoint is added to the Phase 8 FastAPI surface (health, gossip, leader,
expert, generate).

The REPL needs a roster of candidate node addresses. It does **not** assume a fixed
leader/port (the leader moves, and its host port is a convention, not a contract):

1. Default roster is derived from `startup_config.json` (node names) plus the
   `EXTERNAL_PORT_BASE` port convention (leader = 9000, then 9001, …). Overridable via
   `MIXUP_REPL_NODES="host:port,host:port,…"`.
2. Probe candidates in parallel with a short timeout; accept the answer with the **highest
   `epoch`** seen (not just the first non-null reply), and connect to that leader.
3. On any connection error, timeout, or a returned epoch the client has not seen, **re-run
   discovery**. A leader crash mid-session is therefore invisible beyond a short stall.

Picking the highest epoch (rather than the first reply) is what keeps the REPL correct during
a split-brain window — it follows the same "highest epoch wins" rule as Phase 3, and lets the
loser demote on its own.

## Wire contract (text-only, in `mixup-core`)

The prompt moves from a boot-time env var to the request body, so the REPL can send any
prompt without restarting anything (ADR-014's CLI flags become the REPL's flags):

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

class GenerateResponse(BaseModel):
    text: str
    tokens: int            # tokens actually produced
    kv_seq: int            # ledger seq after this generation (diagnostics)
```

These live in `mixup-core` (torch-free, text/int only) so the contract is shared exactly like
the config/types today. The leader's `POST /generate` accepts `GenerateRequest` and returns
`GenerateResponse`.

## Run modes

```
mixup-repl "What is mixture of experts?"     # one-shot, prints the answer
mixup-repl                                   # interactive: prompt → answer loop
```

Interactive loop: read a line, send, print the answer. `:history` shows the client-side
conversation, `:reset` clears it, `:quit` exits. Each prompt re-sends the full history (v1);
see open questions for KV reuse.

## Package & placement

A fourth workspace member `packages/mixup-repl` (torch-free, like `mixup-core`), joining the
dev shell. The alternative — a CLI entry inside `mixup-orchestrator` — is rejected by default
because the REPL is a *consumer* of the running cluster, not an orchestrator concern; keeping
it separate mirrors ADR-001's dependency isolation (a REPL must never drag in `docker` or
`torch`).

## Tunables (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `MIXUP_REPL_NODES` | derived from `startup_config.json` | `host:port` roster to probe |
| `MIXUP_REPL_DISCOVER_TIMEOUT` | 1.0 s | per-node probe timeout |
| `MIXUP_REPL_TIMEOUT` | 60 s | generate deadline (must exceed `MIXUP_EXPERT_TIMEOUT` × layers) |

## Acceptance criteria

- From a clean host, `mixup-repl "..."` returns a correct completion by discovering the
  leader on its own (no `--leader` flag, no fixed port).
- `docker stop` the leader mid-session: the REPL re-discovers and the *next* prompt succeeds
  against the new leader with no manual config change.
- `mixup-repl` never imports torch/docker (CI asserts the dependency stays torch-free).

## Risks

- **Stale leader cache.** If the REPL caches the leader and it dies, the first request fails;
  mitigated by treating *any* connection error as "re-discover."
- **Split-brain window.** The REPL may observe two leaders; it must pick the highest epoch
  and let the loser demote (mirrors Phase 3), not latch onto the first reply.
- **Discovery endpoint is a single node's view.** `GET /leader` reflects one node's possibly
  stale `ClusterView`; the "highest epoch across probes" rule bounds this.

## Open questions

- **Host-side CLI vs. an in-network container?** Default: host-side CLI (simplest for a dev
  REPL). An in-network `mixup-repl` container could use container names and the shared
  roster, at the cost of Docker orchestration.
- **Multi-turn KV reuse?** Default: re-send history each turn (simple, stateless). Reusing
  the leader's KV cache across turns is a future optimization and would need a session id in
  `GenerateRequest`.
- **Streaming?** Deferred (plan non-goal). A `--stream` flag is also the gRPC trigger noted in
  [phase-7-dispatch-resilience.md](phase-7-dispatch-resilience.md).
