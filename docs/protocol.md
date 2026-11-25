# Wire protocol

The central and expert nodes talk over plain HTTP on the Docker bridge network.
Container names are the hostnames, so no IP bookkeeping is needed.

## Addressing and ports

| Constant | Value | Meaning |
|---|---|---|
| `INTERNAL_PORT` | `8000` | every node listens here *inside* the cluster |
| `EXTERNAL_PORT_BASE` | `9000` | central → `9000` on the host; experts → `9001, 9002, …` |

`address(container)` returns `http://{container}:{INTERNAL_PORT}` — e.g.
`http://container_2:8000`. The expert node's FastAPI server binds `0.0.0.0:8000`.

## Endpoints

### `POST /{layer}` (expert node)

The central sends one request per layer to each container that holds a chosen expert.

**Body** — safetensors bytes with three tensors (`ExpertRequest`):

| Key | dtype | shape |
|---|---|---|
| `hidden_states` | bfloat16 | `[tokens, d]` |
| `top_k_index` | int64 | `[tokens, k]` |
| `top_k_weights` | bfloat16 | `[tokens, k]` |

**Response** — safetensors bytes with one tensor (`ExpertResponse`):

| Key | dtype | shape |
|---|---|---|
| `out` | bfloat16 | `[tokens, d]` — this container's partial sum |

`layer` is the path parameter and is passed straight to `ExpertGroup.forward`.

## Why safetensors, not JSON

Tensors are sent as **safetensors** (a typed, layout-stable binary format) rather than
JSON or `pickle`:

- it's far more compact than JSON-encoded floats;
- unlike `pickle`, it can't execute code on load, so the expert node can trust
  arbitrary POST bodies.

The pydantic models `ExpertRequest` / `ExpertResponse` (in `mixup_node.serialization`)
document the shapes and carry `to_bytes()` / `from_bytes()` methods backed by
safetensors; they use `arbitrary_types_allowed=True` so they can hold `torch.Tensor`
fields.

## The central's HTTP client (`Dispatcher`)

- A `ThreadPoolExecutor` with `max_workers = len(expert_containers)` lets the central
  contact every relevant container **in parallel** per layer.
- `dispatch(layer, …)` computes `chosen = {lut[e][0] for e in top_k_index.flatten()}`
  — the set of containers that hold at least one chosen expert for this batch — and
  posts to exactly those.
- It then sums the replies: `total = Σ container.out`.

## Readiness (`wait_for_experts`)

The central blocks before generating, polling each expert container with a `GET`
(timeout 2 s) until it answers. Any HTTP response — even a 404 — means the server is
up (FastAPI is listening), so the loop breaks on the first reply and retries on
connection errors only.

## Concurrency & failure semantics

- Experts are **stateless** across requests: each request carries the full tensors
  needed for that layer, so requests can be retried or load-balanced freely.
- There is no batching across tokens here: the reference implementation generates with
  `do_sample=False` (greedy), so `tokens == 1` per request — the simplest correct case.
- If an expert is unreachable, `requests` raises and generation fails; there is no
  retry/backoff beyond `wait_for_experts` at startup.
