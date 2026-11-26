# Packages

A uv **workspace** of three packages under `packages/`. The split is by *role and
dependency weight*, not by layering.

```
packages/
├── mixup-core/            # the shared contract (torch-free)
│   └── src/mixup_core/
│       ├── types.py        # Config, aliases, ContainerType (StrEnum)
│       ├── config.py       # load_config_file (cluster layout from file)
│       ├── env.py          # NodeEnv model + MIXUP_NODE_ENV (de)serialization
│       ├── constants.py    # ports, network, model paths, address()
│       ├── topology.py     # build_lut, validate_partition, experts_for
│       └── smoketest.py    # `mixup-smoketest` CLI
├── mixup-node/            # in-container runtime
│   └── src/mixup_node/
│       ├── model.py        # ExpertSlot, ExpertGroup, load_central, empty_*, SKIP
│       ├── serialization.py# ExpertRequest/ExpertResponse (safetensors)
│       ├── dispatch.py     # Dispatcher (central → experts HTTP client)
│       ├── server.py       # FastAPI expert server
│       └── infer.py        # `mixup-node` entrypoint (central/expert branch)
└── mixup-orchestrator/    # host tools
    └── src/mixup_orchestrator/
        ├── distribute.py   # `mixup-orchestrator` (Docker build + start)
        └── fetch.py        # `mixup-fetch-model` (Hugging Face download)
```

## Dependency graph

```
mixup-core       → pydantic
mixup-node       → mixup-core + torch, transformers, safetensors, fastapi, uvicorn, requests, pydantic
mixup-orchestrator → mixup-core + docker, huggingface_hub
```

The guiding rule: **`mixup-core` is torch-free.** Anything touching `torch` /
`transformers` / tensors lives in `mixup-node`. This means the orchestrator — which
runs on the host and only talks to Docker — never transitively pulls a 500 MB torch
wheel.

## What goes where

| Concern | Package | Why |
|---|---|---|
| `Config`, aliases, enums | core | both host and node parse it |
| env-var names, ports, network, model paths, `address()` | core | host and node must agree |
| LUT / partition validation | core | pure function of `Config`, unit-testable without torch |
| model construction, expert slicing, `SKIP` | node | torch/transformers internals |
| wire types / safetensors | node | tensors |
| HTTP client / server | node | container-to-container |
| Docker orchestration, model download | orchestrator | host-only |

## Console scripts

| Script | Entry point | Package |
|---|---|---|
| `mixup-smoketest` | `mixup_core.smoketest:main` | core |
| `mixup-node` | `mixup_node.infer:start` | node |
| `mixup-orchestrator` | `mixup_orchestrator.distribute:main` | orchestrator |
| `mixup-fetch-model` | `mixup_orchestrator.fetch:main` | orchestrator |

## Build backend

All three packages use **`uv_build`** (uv's own PEP 517 backend). See
[decisions.md](decisions.md) for why not hatchling.
