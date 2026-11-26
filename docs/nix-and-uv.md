# Nix & uv2nix

`mixup` is developed on NixOS, where bare `uv` does not work on its own. This page
explains why and how the Nix setup works.

## The problem

uv manages its own Python interpreters. On NixOS those interpreters are generic glibc
binaries, and NixOS refuses to run generic glibc executables (no global
`/lib/ld-linux.so`; the "stub-ld" issue). So `uv lock` fails with:

```
Could not start dynamically linked executable: …/python3.15
NixOS cannot run dynamically linked executables intended for generic linux environments
```

There is also no system `python3` on `PATH` by default.

## The solution: uv2nix

`flake.nix` uses [uv2nix](https://github.com/pyproject-nix/uv2nix) (which wraps
[pyproject.nix](https://github.com/pyproject-nix/pyproject.nix)) to build a Nix-native
Python 3.13 environment from `pyproject.toml` + `uv.lock`.

Inputs (all follow `nixpkgs`): `pyproject-nix`, `uv2nix`, `pyproject-build-systems`.

The core pieces of the flake:

- `workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; }` — reads the
  lock and the three members.
- `overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; }` — turns the
  lock into a Nix package set, preferring wheels.
- `editableOverlay = workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; }` —
  makes the three members importable from source (no rebuild on code edits).
- `pythonSets` — `pkgs.python313` + `pyproject-build-systems.overlays.wheel` + the
  workspace overlay.
- `devShells.default` — a `mkVirtualEnv` of `workspace.deps.all` **minus `mixup-node`**
  (see ADR-006), plus `pkgs.uv`.

### The dev-shell env

```nix
env = {
  UV_NO_SYNC = "1";                    # never let uv install its own venv
  UV_PYTHON = pythonSet.python.interpreter;  # point uv at the Nix python
  UV_PYTHON_DOWNLOADS = "never";        # never let uv download an interpreter
};
```

`UV_NO_SYNC=1` is the key one: it makes `uv lock` (and any `uv` command) use the Nix
environment and skip syncing — this is the "nosync" env var referenced throughout the
project.

## Why `mixup-node` is excluded from the dev shell

The node depends on the CUDA torch wheel. Building it under Nix trips the
`nvidia-cufile` package, whose `libcufile_rdma.so` links against RDMA libraries NixOS
doesn't ship (`libmlx5.so.1`, `librdmacm.so.1`, `libibverbs.so.1`), so `auto-patchelf`
fails. The node runs in Docker instead (see [running.md](running.md)), so the dev shell
only needs the host-side tools.

## The lockfile

`uv.lock` is regenerated inside the dev shell (`nix develop` → `uv lock`). It pins the
**CUDA** torch wheel (`torch 2.15.0` + `nvidia-*`/`triton`) plus the workspace members.
The `dev` dependency group (`ruff`, `pytest`) is declared in `pyproject.toml`; note
uv2nix's `workspace.deps.all` does not surface it, which is why `ruff`/`pytest` are not
in the dev shell (run them via `uvx` or a throwaway venv — see
[development.md](development.md)).

## Docker does *not* use Nix

The container image (`Dockerfile`) is a plain `python:3.13` and installs the workspace
with `uv sync --frozen --no-dev --package mixup-node` (pip/uv, not Nix), pulling the CPU
torch wheel from the lock. The NixOS patchelf problem is host-only and, with CPU torch,
irrelevant anyway.
