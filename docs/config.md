# Config

The cluster layout is a single JSON document, read from a file by the orchestrator and
passed (serialized) to every node as the `CONFIG` env var.

## Schema (`mixup_core.types.Config`)

```json
{
  "central": "container_1",
  "experts": {
    "container_2": [0, 1, 2, 3, 4, 5, 6, 7],
    "container_3": [8, 9, 10, 11, 12, 13, 14, 15],
    "container_4": [16, 17, 18, 19, 20, 21, 22, 23],
    "container_5": [24, 25, 26, 27, 28, 29, 30, 31]
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `central` | `str` | Container that runs everything except the experts. |
| `experts` | `dict[str, list[int]]` | Expert container name → the expert indices it holds. |

Type aliases (from `mixup_core.types`): `ContainerName = str`, `ExpertId = int`,
`ExpertList = list[int]`, `ExpertAlloc = dict[str, list[int]]`,
`ExpertRoute = tuple[str, int]` (container, local slot).

## Pydantic validation (core-level, model-agnostic)

`Config` rejects, at parse time:

- **Duplicate experts** — the same expert id appearing on more than one container
  (a `field_validator` on `experts`).
- **Central also in `experts`** — the central holds no experts (a `model_validator`).

These are the invariants that hold regardless of which model is used.

## Node-level validation (model-aware)

On startup, the node additionally runs `validate_partition(config, num_experts)` with
`num_experts` from the model config (32). It rejects:

- **Incomplete coverage** — some expert in `0..31` assigned to no container.
- **Non-contiguous blocks** — a container's experts not forming one ascending run.

See [routing.md](routing.md) for why.

## `role()`

`Config.role(name)` returns `ContainerType.CENTRAL` if `name == central`, else
`ContainerType.EXPERT`. `ContainerType` is a `StrEnum` (`"central"` / `"expert"`).
The node uses this to decide its behavior.

## Environment variables

| Env var | Who reads it | Meaning | Default |
|---|---|---|---|
| `CONFIG` | node | the serialized `Config` | (required) |
| `CONTAINER_NAME` | node | this container's name | (required) |
| `PROMPT` | central | the generation prompt | `"What is mixture of experts?"` |
| `MAX_NEW_TOKENS` | central | how many tokens to generate | `50` |

The orchestrator sets `CONFIG` and `CONTAINER_NAME` on every container, and forwards
`PROMPT` / `MAX_NEW_TOKENS` from the host environment when present (so
`MAX_NEW_TOKENS=1 mixup-orchestrator` produces a fast smoke test).

## Model constants (`mixup_core.constants`)

| Constant | Value | Notes |
|---|---|---|
| `MODEL_ROOT` | `models` | relative to the container `WORKDIR` (`/usr/local/app`) |
| `MODEL_NAME` | `granite-3.1-1b-a400m-instruct` | local dir name |
| `HF_MODEL_ID` | `ibm-granite/granite-3.1-1b-a400m-instruct` | Hugging Face repo |
| `MODEL_DIR` | `models/granite-3.1-1b-a400m-instruct` | |
| `MODEL_FILE` | `models/…/model.safetensors` | the single consolidated safetensors |

The host `models/` directory is mounted read-only into each container at
`/usr/local/app/models`; `MODEL_ROOT` is relative to that mount.
