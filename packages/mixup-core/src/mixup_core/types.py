"""Shared types for the mixup distributed MoE cluster.

This module is the contract both the host-side orchestrator and the in-container
nodes depend on, so it stays free of torch/docker/fastapi: pure pydantic.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

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


class StartupConfig(BaseModel):
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
    def _leader_not_expert(self) -> "StartupConfig":
        """The leader holds everything except the experts, so it holds none."""
        if self.initial_leader in self.experts:
            raise ValueError(f"leader {self.initial_leader!r} is also an expert node")
        return self

    def initial_role(self, name: ContainerName) -> NodeRole:
        """The role `name` starts in."""
        return NodeRole.LEADER if name == self.initial_leader else NodeRole.EXPERT
