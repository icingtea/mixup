"""Loading the startup layout from a file (used by the host-side orchestrator).

The node-side runtime environment lives in `env.py`; this module is only the file
loader for the static startup layout. It accepts either a flat ``StartupConfig`` (the
classic ``startup_config.json``) or a ``LabConfig`` (``lab_config.json``, whose ``startup``
field is extracted), so the gateway and REPL can point at either.
"""

from pathlib import Path

from pydantic import ValidationError

from .types import StartupConfig


def load_startup_config(path: str | Path) -> StartupConfig:
    """Read the initial cluster layout from a JSON file (flat or lab config)."""
    text = Path(path).read_text()
    try:
        return StartupConfig.model_validate_json(text)
    except ValidationError:
        from .lab import LabConfig

        return LabConfig.model_validate_json(text).startup

