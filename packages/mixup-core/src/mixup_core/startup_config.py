"""Loading the startup layout from a file (used by the host-side orchestrator).

The node-side runtime environment lives in `env.py`; this module is only the file
loader for the static startup layout.
"""

from pathlib import Path

from .types import StartupConfig


def load_startup_config(path: str | Path) -> StartupConfig:
    """Read the initial cluster layout from a JSON file."""
    return StartupConfig.model_validate_json(Path(path).read_text())
