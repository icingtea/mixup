"""The node runtime environment: one typed model, one environment variable.

The orchestrator builds a `NodeEnv`, serializes it to JSON, and injects it into each
container as the single `MIXUP_NODE_ENV` env var. The node deserializes it at bootup.
All mixup environment variables live in this one model, so there is a single source
of truth for the runtime contract between host and node.
"""

import os

from pydantic import BaseModel

from .types import Config, ContainerName

NODE_ENV_VAR = "MIXUP_NODE_ENV"
"""The single environment variable carrying a serialized NodeEnv for a container."""

DEFAULT_PROMPT = "What is mixture of experts?"
DEFAULT_MAX_NEW_TOKENS = 50


class NodeEnv(BaseModel):
    """The complete runtime environment handed to a node container."""

    container_name: ContainerName
    """This container's own name (its hostname on the Docker network)."""

    config: Config
    """The cluster layout: which container is central, which experts each holds."""

    prompt: str = DEFAULT_PROMPT
    """The generation prompt (used by the central node)."""

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    """How many tokens the central node generates."""


def load_node_env() -> NodeEnv:
    """Deserialize the MIXUP_NODE_ENV env var into a typed NodeEnv."""
    return NodeEnv.model_validate_json(os.environ[NODE_ENV_VAR])


def dump_node_env(env: NodeEnv) -> str:
    """Serialize a NodeEnv to the JSON string carried by MIXUP_NODE_ENV."""
    return env.model_dump_json()
