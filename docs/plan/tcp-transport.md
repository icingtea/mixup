# Raw TCP expert transport (lever 2)

Replaces the HTTP *data plane* (the per-layer expert call) with a persistent raw-TCP
binary protocol. The HTTP *control plane* (health / leader / routing / join / generate)
stays on FastAPI. This removes the per-request FastAPI/uvicorn + requests/urllib3
overhead, which the ledger shows as ~1.7 ms of a ~4.4 ms round-trip.

## Protocol

- Frame: `[u32 big-endian length][payload]`.
- Request payload (leader -> expert): `[u32 big-endian layer][safetensors body]`.
- Response payload (expert -> leader): `[f64 big-endian server_seconds][safetensors body]`.

The tensor payload keeps the existing safetensors format (`ExpertRequest`/`ExpertResponse`)
unchanged — only the transport framing around it changes, so correctness risk is low.

## Codepaths

| File | Change |
|------|--------|
| `packages/mixup-core/src/mixup_core/constants.py` | add `EXPERT_PORT = 8001` (internal data plane) |
| `packages/mixup-node/src/mixup_node/transport.py` | **new** — `send_frame`/`recv_frame`/`recv_exact`, `TCPConnection` (client), `TCPServer` (server). Torch-free. |
| `packages/mixup-node/src/mixup_node/dispatch.py` | `_send` uses a per-node `TCPConnection` instead of `requests.Session.post` |
| `packages/mixup-node/src/mixup_node/infer.py` | start a `TCPServer` on 8001 in a thread; callback runs `expert_group.forward` |
| `tests/test_transport.py` | **new** — framing + round-trip + partial reads + reuse + failure (torch-free) |

## Connection model

- Leader (`Dispatcher`) holds one persistent `TCPConnection` per expert node, lazily
  connected and reused. Generation is sequential (one layer at a time, one job per node),
  so each connection is used by at most one thread at a time — no per-socket locking.
- On send/recv failure: close + drop the connection, raise `_NodeFailure` (the existing
  failover path already handles this).

## Server model

- Every node binds its `TCPServer` on `:8001` **synchronously at boot**, before experts
  load, so it is listening by the time `/health` reports `ready`.
- Accept loop + one daemon thread per connection; each connection serves a serial stream
  of request/response frames (matching the leader's one-connection-per-node model).
- Callback guards `expert_group is None` (leader) by returning an error frame.

## Expected impact

- Network slice of the round-trip: ~1.74 ms -> ~0.3 ms.
- Round-trip ~4.4 ms -> ~3.0 ms; dispatch ~6.1 s -> ~4.4 s; ~6.2 -> ~7.5 tok/s (~20%).
- (The server's ~1.9 ms of GPU transfer/sync is *not* affected by transport — that is
  the remaining floor, and it is addressed separately.)

## Critique / risks

1. **No concurrency benefit, and that's fine.** Serial generation means we never overlap
   requests on a node, so a single persistent socket is correct. If batching/pipelining is
   added later (lever 1 is a NO GO, but batch-prefill isn't), the connection model needs
   a per-connection lock — noted, not done now.
2. **Rebalance vs in-flight forward.** `set_routing` swaps `self.layers` while a forward
   could be running. Pre-existing race (the HTTP path had it too); the forward holds a
   reference to the old layer list, so it completes against the old weights. Acceptable.
3. **Stale connections.** On rebalance a removed node's connection lingers; close it in
   the failure/rebalance path to avoid a leak.
4. **Port 8001 is internal-only.** Containers reach each other by name over the bridge;
   no host port mapping or orchestrator change needed.
5. **Metric field names.** `http_seconds` now means "transport round-trip"; kept the name
   so existing JSONL analysis stays valid.
