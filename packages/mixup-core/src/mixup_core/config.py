"""Loading and validating the cluster Config, from a file (host) or the environment (node)."""

import os
from pathlib import Path

from .types import Config

CONFIG_ENV = "CONFIG"
CONTAINER_NAME_ENV = "CONTAINER_NAME"
PROMPT_ENV = "PROMPT"
MAX_NEW_TOKENS_ENV = "MAX_NEW_TOKENS"


def load_config_file(path: str | Path) -> Config:
    """Read the cluster layout from a JSON file. Used by the orchestrator."""
    return Config.model_validate_json(Path(path).read_text())


def load_config_from_env() -> Config:
    """Read the cluster layout from the CONFIG environment variable. Used inside a node."""
    return Config.model_validate_json(os.environ[CONFIG_ENV])


def container_name_from_env() -> str:
    """The node's own container name, from CONTAINER_NAME."""
    return os.environ[CONTAINER_NAME_ENV]
