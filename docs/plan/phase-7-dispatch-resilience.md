# Phase 7 — Dispatch resilience & transport

**Goal:** make the leader's expert fan-out survive transient failures, and settle the
transport (HTTP vs gRPC) with a concrete, low-cost improvement path.

## Dispatch resilience

Today's `Dispatcher` (see `mixup_node/dispatch.py`) uses a static `self.lut` and
`self.containers`, opens a **fresh TCP connection per request** (`requests.post` with no
`Session`), and has no retry or deadline. This phase rewrites it:

1. **Live routing table.** The dispatcher reads the current `RoutingTable` (Phase 6)
   instead of `self.lut`. A request pins the table version it was issued under.
2. **Persistent connection pooling.** Replace per-call `requests.post` with a module-level
   `requests.Session` (keep-alive) or `httpx.Client`. This eliminates thousands of TCP
   handshakes per generation — the dominant protocol cost today.
3. **Deadlines.** Every request carries an explicit timeout (`MIXUP_EXPERT_TIMEOUT`,
   default e.g. 30 s, comfortably above one layer's compute). gRPC-style deadlines are
   emulated with a `timeout` + a `deadline` header.
4. **Retry & re-route.** On a connection error/timeout, mark the node suspect (feed the
   suspicion to gossip), and retry against the **new owner** from the latest table, up to
   `MIXUP_MAX_RETRIES`. Because expert requests are stateless/idempotent (they carry the
   full tensors), a retry to a *different* node is safe.
5. **Partial-sum integrity.** The leader sums partials from the owners it contacted; if a
   re-route changes owners mid-layer, the retry is issued to the authoritative owner of
   the *un-answered* experts only, so the sum stays exact (no double-count, no gap).

### Determinism of the reduction (accepted drift)

Greedy decoding is deterministic *given a fixed routing table*, but the reduce `y = Σ_j y_j`
sums partials *over containers*, and failover rebalances which experts live on which
container — changing the grouping/order of the float sum and moving the result by ~1 ULP.
When the top two logits are near-tied that can flip an `argmax` and, because generation is
autoregressive, change the tokens after it.

This is accepted, not fixed. Failover inherently changes placement, and a tiny numeric
difference is a property of floating-point arithmetic, not a defect. The one cheap,
worthwhile step is **order-stable accumulation** — sum partials sorted by container name —
so repeated runs under an *unchanged* table are bit-identical and the no-failure baseline
is reproducible. Per-expert (expert-id-ordered) partials would make the sum independent of
placement, but that is a wire-format change and a non-goal.

The chaos oracle therefore does **not** demand token-identical output after a failover; it
asserts generation completes correctly (valid, terminating output; no double-count; no
crash). See [testing.md](testing.md).

## Transport decision: stay on HTTP/FastAPI

Weighing gRPC vs the existing HTTP+safetensors stack, the recommendation is to **stay on
HTTP/FastAPI**, with the pooling + deadline changes above. The reasoning:

- **The dominant payloads are already binary.** Expert request/response tensors are
  safetensors bytes today; gRPC would carry the *same* bytes in a protobuf `bytes` field,
  so its serialization advantage is nil on the hot path ([gRPC], [safetensors]).
- **The bottleneck is compute + sequential layer latency, not protocol.** Each token does
  24 sequential layer round-trips; each round-trip is dominated by the expert matmul. The
  protocol overhead is sub-millisecond noise ([Switch], [Mixtral]).
- **gRPC's wins are real but marginal here:** HTTP/2 multiplexing helps concurrent
  fan-out (we already fan out over a thread pool), deadlines/retries we're adding at the
  app layer, and codegen typing duplicates the pydantic contract we already have
  ([gRPC], [HTTP2]).
- **gRPC's costs are concrete in this repo:** a compiled `grpcio` + `grpcio-tools`/protoc
  codegen build step is a new vector for the exact Nix/uv2nix native-dep failures we've
  been dodging (see `docs/nix-and-uv.md`), and it worsens the curl-based debug story.

The **one future trigger** that would justify gRPC is *streaming* — continuous token
output to a client, or a high-rate bidirectional control plane. Even then, we'd introduce
gRPC on that plane only and keep the expert data path on HTTP+safetensors where it is
already optimal. (Also noted: speculative decoding would change the request pattern, but
is out of scope, [SpecDec].)

> **Streaming/sampling breaks recovery.** The KV-ledger recovery design (Phase 5) assumes
> deterministic greedy decoding. Streaming or sampling changes the request pattern *and*
> invalidates "recompute the tail exactly"; any of those future directions must revisit
> the recovery design at the same time.

### FastAPI vs. a leaner server

FastAPI is retained: the pydantic request/response models and OpenAPI surface align with
the project's pydantic-first standard. If the concern is per-request overhead rather than
the protocol, the fix is the connection pool, not a server swap. (A pure-`uvicorn`/`httpx`
data plane is a possible micro-optimization if profiling ever shows FastAPI routing
overhead — not expected at this scale.)

## The transport as a separate layer

The gossip/heartbeat/routing endpoints (Phase 2) and the expert endpoint share one
FastAPI app but must be **physically isolated**, not just logically separate: a slow
expert matmul or request storm must not starve heartbeats into a false `SUSPECT`/`DEAD`
(and a needless rebalance). Concretely:

- The gossip transport runs on its **own executor** (a dedicated `ThreadPoolExecutor` or
  asyncio task group), separate from the expert fan-out pool.
- Heartbeats are **fire-and-forget with a wall-clock deadline** measured at send time, so
  request load can't make a heartbeat appear late.
- The resilient `Dispatcher` is the only client of the expert path.

This keeps the data path (hot, binary, pooled) cleanly separated from the control path
(small, JSON, gossip), and the control path isolated from data-path load.

## Tunables

| Var | Default | Meaning |
|---|---|---|
| `MIXUP_EXPERT_TIMEOUT` | 30 s | deadline for one expert request |
| `MIXUP_MAX_RETRIES` | 2 | re-route attempts per layer |
| `MIXUP_HTTP_POOL_SIZE` | 16 | connection pool size |
| `MIXUP_HTTP2` | false | use `httpx` with HTTP/2 if set |

## Acceptance criteria

- A single dropped expert request does not abort generation (retry/re-route succeeds).
- Connections are reused (assert via `docker stats`/net or a connection-count probe).
- Request storms against a table version bump never double-count an expert.
- Decision record (ADR) captures "stay on HTTP; gRPC deferred to a streaming trigger."

## Risks

- **Retry amplification** under load. Mitigation: bounded retries + a per-node circuit
  breaker (open after `MIXUP_MAX_RETRIES`, half-open after a backoff) — documented, can be
  a fast follow.
- **Deadline vs. slow matmul.** A too-tight deadline causes spurious retries. Mitigation:
  conservative default + metric-driven tuning later.
