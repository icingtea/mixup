"""Shared types for the mixup distributed MoE cluster.

This module is the contract both the host-side orchestrator and the in-container
nodes depend on, so it stays free of torch/docker/fastapi: pure pydantic.
"""

from enum import StrEnum

from pydantic import field_validator, model_validator

from .base import StrictModel

ContainerName = str
"""A node's name, which is also its hostname on the Docker network, e.g. "container_2"."""

NodeId = str
"""A node's unique id (its container name / hostname)."""

Epoch = int
"""A monotonic logical clock of leadership generations, incremented each election."""

NodePriority = int
"""An election tie-break; higher wins."""

Incarnation = int
"""A node's boot counter, incremented each restart, so a restarted node is distinguishable."""

ExpertId = int
"""A global expert index within a single layer, 0 .. num_local_experts-1."""

ExpertList = list[ExpertId]
"""The experts held by one node."""

ExpertAlloc = dict[ContainerName, ExpertList]
"""Every expert node and the experts it holds."""

ExpertRoute = tuple[ContainerName, int]
"""Where an expert lives: (node name, its local slot index on that node)."""


class NodeEndpoint(StrictModel):
    """How the rest of the cluster reaches one node, for cross-host deployments.

    ``http`` is the control-plane base URL (no trailing slash); ``expert_host`` and
    ``expert_port`` are the raw-TCP expert data plane. When a node has no entry in
    ``StartupConfig.endpoints``, the single-host Docker defaults apply (the container
    name on the Docker network, ports ``INTERNAL_PORT`` / ``EXPERT_PORT``).
    """

    http: str
    """Control-plane base URL, e.g. ``http://100.102.170.72:9000``."""

    expert_host: str
    """Host for the raw-TCP expert data plane, e.g. ``100.102.170.72``."""

    expert_port: int = 8001
    """Published port for the expert data plane (defaults to the in-cluster port)."""

    @field_validator("http")
    @classmethod
    def _http_scheme(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"endpoint http must be an http(s) URL, got {value!r}")
        return value

    @field_validator("expert_host")
    @classmethod
    def _expert_host_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("endpoint expert_host must not be empty")
        return value


class NodeRole(StrEnum):
    """What role a node plays in the cluster.

    A role is dynamic at runtime: an expert node can be promoted to leader. The
    startup layout only fixes the *initial* role (see ``StartupConfig.initial_role``).
    """

    LEADER = "leader"
    EXPERT = "expert"


class MemberStatus(StrEnum):
    """Liveness of a member, per the SWIM failure detector (see gossip.py)."""

    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


class StartupConfig(StrictModel):
    """The *initial* cluster layout, read from startup_config.json on the host.

    This describes only the boot-time arrangement: which node starts as leader and
    which experts each node initially holds. The live layout is a separate
    ``RoutingTable`` (not this type), which failover and rebalancing mutate.
    """

    initial_leader: ContainerName
    """The node that starts as the generation leader."""

    nodes: list[ContainerName]
    """The full membership roster (leader + expert nodes), for peer discovery."""

    experts: ExpertAlloc
    """The initial expert assignment (the leader holds none)."""

    endpoints: dict[ContainerName, NodeEndpoint] = {}
    """Per-node reachability for cross-host deployments.

    Empty for the single-host lab, in which case nodes fall back to Docker container
    names. When non-empty it must map every node in ``nodes`` (and nothing else).
    """

    kv_sync_every: int = 5
    """How often (in generated tokens) the leader broadcasts its warm KV cache to peers."""

    predictor_kind: str = "mlp"
    """The next-layer predictor to train: ``"lr"`` (logistic regression) or ``"mlp"``
    (CoMoE's 2-layer MLP, the default)."""

    max_new_tokens: int = 500
    """Default generation length when a client doesn't specify one."""

    temperature: float = 0.7
    """Sampling temperature (0 = greedy)."""

    top_k: int = 50
    """Top-k sampling cutoff (0/1 = greedy)."""

    @field_validator("experts")
    @classmethod
    def _experts_unique(cls, value: ExpertAlloc) -> ExpertAlloc:
        """Each expert id may be assigned to at most one node."""
        seen: set[ExpertId] = set()
        for experts in value.values():
            for expert in experts:
                if expert in seen:
                    raise ValueError(f"expert {expert} is assigned to more than one node")
                seen.add(expert)
        return value

    @model_validator(mode="after")
    def _endpoints_cover_roster(self) -> "StartupConfig":
        """If endpoints are given, they must cover exactly the roster (no more, no less)."""
        if not self.endpoints:
            return self
        roster = set(self.nodes)
        endpoint_nodes = set(self.endpoints)
        missing = sorted(roster - endpoint_nodes)
        extra = sorted(endpoint_nodes - roster)
        if missing or extra:
            raise ValueError(
                f"endpoints must match nodes exactly: missing={missing} extra={extra}"
            )
        return self

    def initial_role(self, name: ContainerName) -> NodeRole:
        """The role `name` starts in."""
        return NodeRole.LEADER if name == self.initial_leader else NodeRole.EXPERT


class GenerateRequest(StrictModel):
    """A generation request sent to the leader's ``POST /generate``."""

    prompt: str
    max_new_tokens: int = 500


class GenerateResponse(StrictModel):
    """The leader's reply to a generation request, with benchmark metrics.

    ``dense_seconds`` is the leader's own (non-distributed) compute; ``dispatch_seconds``
    is the distributed expert dispatch + reduce time.
    """

    text: str
    tokens: int
    elapsed_seconds: float
    dense_seconds: float
    dispatch_seconds: float
