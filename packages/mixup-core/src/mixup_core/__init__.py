"""mixup-core: the shared contract for the distributed MoE cluster.

Types, config, constants, routing, and the node runtime environment shared by
mixup-orchestrator (host) and mixup-node (in-container). Deliberately
torch/docker/fastapi-free.
"""

from .constants import (
    CHECKPOINT_FILE,
    EXTERNAL_PORT_BASE,
    HF_MODEL_ID,
    IMAGE_NAME,
    IMAGE_TAG,
    INTERNAL_PORT,
    LEDGER_DIR,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_ROOT,
    NETWORK_NAME,
    address,
)
from .election import (
    apply_leader_announcement,
    current_leader,
    elect,
    leader_projection,
    promote,
    should_become_leader,
)
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
from .kvledger import (
    Checkpoint,
    LedgerHeader,
    decode_checkpoint,
    decode_header,
    encode_checkpoint,
    encode_header,
)
from .runtime import (
    ClusterView,
    LeaderProjection,
    LoadMetrics,
    MemberState,
    RoutingEntry,
    RoutingTable,
    apply_routing,
    expert_moves,
)
from .scheduler import (
    LeastLoadedPolicy,
    PlacementPolicy,
    handle_node_failure,
    rebalance_experts,
)
from .startup_config import load_startup_config
from .topology import (
    build_lut,
    cluster_view_from_config,
    expert_containers,
    experts_for,
    routing_table_from_config,
    validate_partition,
)
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
    "CHECKPOINT_FILE",
    "LEDGER_DIR",
    "MODEL_DIR",
    "MODEL_FILE",
    "MODEL_NAME",
    "MODEL_ROOT",
    "NETWORK_NAME",
    "address",
    "build_lut",
    "cluster_view_from_config",
    "dump_node_env",
    "expert_containers",
    "experts_for",
    "load_startup_config",
    "load_node_env",
    "routing_table_from_config",
    "validate_partition",
    # runtime state
    "ClusterView",
    "LeaderProjection",
    "LoadMetrics",
    "MemberState",
    "RoutingEntry",
    "RoutingTable",
    "apply_routing",
    "expert_moves",
    "elect",
    "leader_projection",
    "current_leader",
    "should_become_leader",
    "promote",
    "apply_leader_announcement",
    "LeastLoadedPolicy",
    "PlacementPolicy",
    "rebalance_experts",
    "handle_node_failure",
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
    "Checkpoint",
    "decode_checkpoint",
    "encode_checkpoint",
]
