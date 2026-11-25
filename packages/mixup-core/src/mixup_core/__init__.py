"""mixup-core: the shared contract for the distributed MoE cluster.

Types, config, constants, and routing shared by mixup-orchestrator (host) and
mixup-node (in-container). Deliberately torch/docker/fastapi-free.
"""

from .config import (
    CONFIG_ENV,
    CONTAINER_NAME_ENV,
    MAX_NEW_TOKENS_ENV,
    PROMPT_ENV,
    container_name_from_env,
    load_config_file,
    load_config_from_env,
)
from .constants import (
    EXTERNAL_PORT_BASE,
    HF_MODEL_ID,
    IMAGE_NAME,
    IMAGE_TAG,
    INTERNAL_PORT,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_ROOT,
    NETWORK_NAME,
    address,
)
from .topology import build_lut, expert_containers, experts_for, validate_partition
from .types import (
    Config,
    ContainerName,
    ContainerType,
    ExpertAlloc,
    ExpertId,
    ExpertList,
    ExpertRoute,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ContainerName",
    "ContainerType",
    "ExpertAlloc",
    "ExpertId",
    "ExpertList",
    "ExpertRoute",
    "CONFIG_ENV",
    "CONTAINER_NAME_ENV",
    "MAX_NEW_TOKENS_ENV",
    "PROMPT_ENV",
    "EXTERNAL_PORT_BASE",
    "HF_MODEL_ID",
    "IMAGE_NAME",
    "IMAGE_TAG",
    "INTERNAL_PORT",
    "MODEL_DIR",
    "MODEL_FILE",
    "MODEL_NAME",
    "MODEL_ROOT",
    "NETWORK_NAME",
    "address",
    "build_lut",
    "container_name_from_env",
    "expert_containers",
    "experts_for",
    "load_config_file",
    "load_config_from_env",
    "validate_partition",
]
