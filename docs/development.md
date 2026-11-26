# Development

## Lint

Ruff, with a workspace config in the root `pyproject.toml` (`[tool.ruff]`):

```bash
ruff check .
```

Selected rules: `E`, `F`, `W`, `I` (isort), `UP` (pyupgrade), `B` (bugbear),
`SIM` (simplify). Line length 100, target `py313`, `src` covers the three package
source dirs, `.temp` is excluded.

## Tests

`pytest` (in `tests/`). The tests cover the pure, torch-free parts:

- `test_config.py` — `StartupConfig` validation (duplicates, leader-in-experts) and loading.
- `test_topology.py` — `build_lut`, `experts_for`, `validate_partition`
  (coverage + contiguity failures).
- `test_env.py` — `NodeEnv` (de)serialization round-trip and defaults.

`pytest` is in the lock's `dev` group but not surfaced into the dev shell by uv2nix
(see [nix-and-uv.md](nix-and-uv.md)), so run tests with a throwaway venv:

```bash
nix shell nixpkgs#python313 --command bash -lc '
  export UV_PYTHON=$(command -v python3)
  uv venv /tmp/mixup-test-venv
  uv pip install --python /tmp/mixup-test-venv/bin/python -q pytest pydantic
  PYTHONPATH="packages/mixup-core/src" /tmp/mixup-test-venv/bin/python -m pytest tests -q
'
```

The topology tests only need `mixup_core` (torch-free) — that's the point of ADR-002.

## Lockfile

Regenerate after changing dependencies:

```bash
nix develop
uv lock          # UV_NO_SYNC=1 is set, so this just writes uv.lock
```

## Conventions

- **Type hints everywhere** (PEP 604 unions, `list[int]`, `StrEnum`, `collections.abc`).
- **Docstrings** on every public module/class/function.
- **`__all__`** in `mixup_core/__init__.py` (the public surface).
- **No module-level side effects** in `mixup-node` — state is passed in
  (`model_config`, `model_file`, `lut`) rather than read from globals.
- **`logging`**, not `print`; **specific exceptions**, not bare `except`.
- **StartupConfig as the single source of truth** for routing (no `// %` heuristics).
- **Pydantic models**, not `TypedDict`, for data contracts.

## Adding an expert node

1. Add its block to `startup_config.json` (`"container_6": [32..39]` etc.) — keep the expert
   lists contiguous and covering the full `0..31` range.
2. Re-run `mixup-smoketest` to confirm the partition.
3. Re-run `mixup-orchestrator` — it starts the new container automatically and the
   central's `ThreadPoolExecutor` is sized from the config, so no code change is needed.

## Known limitations

- **Sharded safetensors** are not handled (needs a single `model.safetensors`).
- **Central restart-loop** (ADR-011): the central is one-shot but has
  `restart_policy: unless-stopped`.
- **No batching** across tokens; greedy decoding only (`do_sample=False`).
- **No retry/backoff** on expert requests beyond startup readiness polling.
- The **transformers internals** (`_experts_implementation`, `_is_expert_parallel`,
  the `router.layer.weight` rename) are private-API fragile (ADR-008).
