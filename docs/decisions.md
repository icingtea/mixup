# Design decisions (ADRs)

Every non-obvious choice, and why. Written as lightweight Architecture Decision Records.

---

## ADR-001: Three packages — `mixup-core`, `mixup-node`, `mixup-orchestrator`

**Decision.** Split the project into three workspace members by role.

**Why.** The original prototype was a single `infer.py` that did everything, plus a
`models.py` that was only imported by the orchestrator — so the container script
re-parsed config as an untyped `dict[str, Any]` and duplicated constants. Three
packages give each role a clear home and, crucially, isolate heavy dependencies:
the orchestrator doesn't need torch, the node doesn't need docker.

**Alternatives.** One package with submodules (simpler, but "core" isn't independently
installable and dependency isolation is lost); a `mixup-core` + single `mixup` app
(the app would still mix orchestrator and node concerns).

---

## ADR-002: `mixup-core` is torch-free

**Decision.** `mixup-core` depends only on `pydantic`. Everything tensor-related lives
in `mixup-node`.

**Why.** The core is the *contract* shared by host and container: types, constants,
config, routing. Routing is a pure function of `StartupConfig`, so it belongs here and is
unit-testable without installing torch. Keeping torch out means the host-side
orchestrator (which pulls in `mixup-core`) never downloads a 500 MB wheel.

**Consequence.** `SKIP` (a transformers runtime sentinel) is deliberately *not* in
core — it lives in `mixup_node.model`.

---

## ADR-003: Load experts by `safe_open` slices, not `from_pretrained`

**Decision.** The node reads `model.safetensors` with `safe_open` and `get_slice`,
never `AutoModelForCausalLM.from_pretrained` for the weights.

**Why.** Two hard requirements conflict with `from_pretrained`:

1. The **central** must load everything *except* the experts and substitute an
   `ExpertSlot` stub per layer.
2. Each **expert** must load *only its own* experts — a contiguous slice of the two
   expert weight tensors — so no node materializes the full model.

`from_pretrained` loads the whole model into memory; slicing avoids that. This is the
core memory win of the design.

**Consequence.** The model must be a *single consolidated* `model.safetensors` file
(not HF shards), and we assume the weight names match transformers' `load_state_dict`
mapping (see ADR-008).

---

## ADR-004: Derive the LUT from `config.experts`, validate at startup

**Decision.** `build_lut` uses the explicit per-container expert lists, and the node
runs `validate_partition` (coverage + contiguity) before loading.

**Why.** The original used `expert = // ` and `slot = % ` over the container count,
ignoring the lists and silently breaking for non-contiguous/uneven partitions. The
explicit LUT makes the config the single source of truth, and validation turns a
subtle mis-routing into a loud startup failure.

---

## ADR-005: uv2nix for the dev environment

**Decision.** Provide a `flake.nix` (uv2nix) dev shell; don't rely on bare `uv`.

**Why.** This is a NixOS machine. uv's managed Python interpreter is a generic glibc
binary that NixOS can't execute (`stub-ld`), so `uv lock` fails out of the box. uv2nix
builds a Nix-native Python 3.13 environment from `uv.lock`. The shell sets
`UV_NO_SYNC=1`, `UV_PYTHON`, and `UV_PYTHON_DOWNLOADS=never` so `uv` uses the Nix env
and never tries to sync its own.

---

## ADR-006: The dev shell excludes `mixup-node` (CUDA)

**Decision.** The dev shell's virtualenv is `workspace.deps.all` minus `mixup-node`.

**Why.** The node runs in Docker (via pip), not Nix. Building it in the dev shell drags
in the CUDA torch wheel, whose `nvidia-cufile` package links against InfiniBand/RDMA
libraries (`libmlx5`, `librdmacm`, `libibverbs`) that NixOS doesn't provide — the
`auto-patchelf` step fails. Excluding the node keeps `nix develop` cheap and correct.

**Consequence.** The dev shell gives you `mixup-smoketest`, `mixup-orchestrator`,
`mixup-fetch-model`, and `uv` — but not the node runtime (which belongs in Docker).

---

## ADR-007: CPU torch wheel (for dev speed; GPU later)

**Decision.** Resolve torch from the `pytorch-cpu` index
(`https://download.pytorch.org/whl/cpu`), not PyPI's default CUDA-bundled wheel.

**Why.** For development, build/startup speed matters more than GPU inference. PyPI's
default `torch` for Linux bundles CUDA 13 and pulls ~18 `nvidia-*` packages plus
`triton` — gigabytes that make every `docker build` slow. The CPU wheel (`torch+cpu`)
drops all of that, so the image is far smaller and builds much faster.

**Consequence.** The lock resolves `torch 2.15.0+cpu` with no `nvidia-*`/`triton`/`cuda`
packages. To re-enable GPU inference later, point `torch` back at PyPI (remove the
`pytorch-cpu` index + the `torch` source) and re-lock. Also note this is exactly what
made the old ADR-006 exclusion moot in principle — though the dev shell still excludes
`mixup-node` because torch (CPU or GPU) isn't needed host-side.

---

## ADR-008: transformers-internal hooks for expert parallelism

**Decision.** In `mixup_node.model`, use three transformers internals:

- `small_config._experts_implementation = "grouped_mm"` — the fast batched-matmul path.
- `experts._is_expert_parallel = True` — enables skipping out-of-range expert slots.
- `key.replace("router.layer.weight", "router.weight")` when loading central weights.

**Why.** These are the mechanisms transformers itself uses for expert-parallel training,
and they're what makes a `g`-sized experts module interpret `SKIP` slots as "skip".
The rename is because the on-disk safetensors uses an older key name than the model
expects. These are the most fragile points (they're private APIs); see ADR-009.

---

## ADR-009: `SKIP = 999` sentinel (belt-and-suspenders masking)

**Decision.** Non-owned experts are masked twice: their routing weight is zeroed, *and*
their local slot is set to `SKIP = 999` (≥ the container's expert count) so the
expert-parallel path skips the compute.

**Why.** The weight-zeroing is semantically sufficient; the `SKIP` slot is an
optimization (skip the matmul entirely) and a safety net against the private
`_is_expert_parallel` behavior changing. Either mechanism alone would be enough, so
the pair is robust.

---

## ADR-010: `uv_build` as the build backend (not hatchling)

**Decision.** Every package declares `build-backend = "uv_build"`.

**Why.** uv2nix builds workspace packages in *editable* mode for the dev shell.
hatchling's editable build imports the `editables` package, which isn't present in the
uv2nix build environment → `ModuleNotFoundError: No module named 'editables'`. uv's own
backend handles editable installs natively and is what the uv2nix docs/reference use.

---

## ADR-011: One-shot central vs. a server

**Status.** Open.

The central currently generates once and exits. Combined with
`restart_policy: "unless-stopped"`, this makes it **restart-loop** (regenerate, exit,
restart). The clean long-term shape is to make the central a persistent
`POST /generate` server (or drop the restart policy on the central only). Documented
here as a known limitation rather than a settled decision.

---

## ADR-012: Safetensors over the wire

**Decision.** Request/response tensors are serialized with safetensors, and the shapes
are documented by pydantic `ExpertRequest` / `ExpertResponse` models.

**Why.** Typed, compact, and (unlike `pickle`) safe to deserialize from an untrusted
network peer. See [protocol.md](protocol.md).

---

## ADR-013: `logging` over `print`, specific exceptions over bare `except`

**Decision.** Nodes and the orchestrator log via the `logging` module (level `INFO`,
`format="%(message)s"` so Docker logs stay clean); exception handlers catch specific
types (`docker.errors.NotFound`, `requests.exceptions.RequestException`) rather than
bare `except:`.

**Why.** Standard library logging gives level control and correct stderr semantics;
specific catches avoid swallowing unrelated bugs.

---

## ADR-014: Prompt / token budget via orchestrator CLI flags

**Decision.** The generation prompt and token budget are fields on `NodeEnv` (with
defaults) and are set via `mixup-orchestrator --prompt ... --max-new-tokens ...`.

**Why.** It makes the forward pass cheaply testable (`--max-new-tokens 1`) without
recompiling or editing the image, and keeps them in the same single model as the rest
of the runtime environment (ADR-015).

---

## ADR-015: One runtime environment model, one env var

**Decision.** All runtime settings a node needs (container name, cluster config,
prompt, token budget) are bundled into one pydantic model, `NodeEnv`
(`mixup_core.env`), serialized to JSON, and injected into each container as a single
env var, `MIXUP_NODE_ENV`. The node deserializes it at bootup via `load_node_env()`.

**Why.** The earlier scheme scattered the environment across four env vars (`CONFIG`,
`CONTAINER_NAME`, `PROMPT`, `MAX_NEW_TOKENS`) whose names were duplicated as magic
strings across two packages, and the node re-parsed them untyped. One model gives a
single source of truth in `mixup-core`, one serialization format, and — with the
`MIXUP_` prefix rule — a clean namespace: the only mixup env var is `MIXUP_NODE_ENV`.

**Consequence.** Generation settings moved from host env vars to CLI flags (ADR-014);
`load_config_from_env` / `container_name_from_env` were removed in favour of
`load_node_env`.
