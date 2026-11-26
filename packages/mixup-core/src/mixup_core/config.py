"""Loading the cluster Config from a file (used by the host-side orchestrator).

The node-side runtime environment lives in `env.py`; this module is only the file
loader for the static cluster layout.
"""

from pathlib import Path

from .types import Config


def load_config_file(path: str | Path) -> Config:
    """Read the cluster layout from a JSON file."""
    return Config.model_validate_json(Path(path).read_text())
