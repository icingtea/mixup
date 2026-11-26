"""mixup-core: the shared contract for the distributed MoE cluster.

Types, config, constants, routing, and the node runtime environment shared by
mixup-orchestrator (host) and mixup-node (in-container). Deliberately
torch/docker/fastapi-free.
"""

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
from .election import elect, leader_projection
from .env import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    NODE_ENV_VAR,
    NodeEnv,
    dump_node_env,
    load_node_env,
)
from .failure_detection import Action, FailureDetector
from .gossip import (
    Ack,
    Alive,
    Dead,
    GossipMsg,
    Ping,
    PingReq,
    Suspect,
    apply_gossip,
    decode_gossip,
    encode_gossip,
    merge_views,
)
from .gossip_engine import GossipEngine
from .kvledger import LedgerHeader, decode_header, encode_header
from .runtime import (
    ClusterView,
    LeaderProjection,
    LoadMetrics,
    MemberState,
    RoutingEntry,
    RoutingTable,
    apply_routing,
)
from .scheduler import LeastLoadedPolicy, PlacementPolicy, rebalance_experts
from .startup_config import load_startup_config
from .topology import build_lut, expert_containers, experts_for, validate_partition
from .types import (
    ContainerName,
    Epoch,
    ExpertAlloc,
    ExpertId,
    ExpertList,
    ExpertRoute,
    Incarnation,
    MemberStatus,
    NodeId,
    NodePriority,
    NodeRole,
    StartupConfig,
)

__version__ = "0.1.0"

__all__ = [
    "ContainerName",
    "NodeRole",
    "MemberStatus",
    "StartupConfig",
    "Epoch",
    "ExpertAlloc",
    "ExpertId",
    "ExpertList",
    "ExpertRoute",
    "Incarnation",
    "NodeId",
    "NodePriority",
    "NodeEnv",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROMPT",
    "NODE_ENV_VAR",
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
    "dump_node_env",
    "expert_containers",
    "experts_for",
    "load_startup_config",
    "load_node_env",
    "validate_partition",
    # runtime state
    "ClusterView",
    "LeaderProjection",
    "LoadMetrics",
    "MemberState",
    "RoutingEntry",
    "RoutingTable",
    "apply_routing",
    "elect",
    "leader_projection",
    "LeastLoadedPolicy",
    "PlacementPolicy",
    "rebalance_experts",
    # failure detection
    "Action",
    "FailureDetector",
    # gossip
    "Ack",
    "Alive",
    "Dead",
    "GossipMsg",
    "Ping",
    "PingReq",
    "Suspect",
    "apply_gossip",
    "merge_views",
    "GossipEngine",
    "decode_gossip",
    "encode_gossip",
    # kv ledger
    "LedgerHeader",
    "decode_header",
    "encode_header",
]
