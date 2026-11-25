"""Shared types for the mixup distributed MoE cluster.

This module is the contract both the host-side orchestrator and the in-container
nodes depend on, so it stays free of torch/docker/fastapi: pure pydantic.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

ContainerName = str
"""A container's name, which is also its hostname on the Docker network, e.g. "container_2"."""

ExpertId = int
"""A global expert index within a single layer, 0 .. num_local_experts-1."""

ExpertList = list[ExpertId]
"""The experts held by one container."""

ExpertAlloc = dict[ContainerName, ExpertList]
"""Every expert container and the experts it holds."""

ExpertRoute = tuple[ContainerName, int]
"""Where an expert lives: (container name, its local slot index on that container)."""


class ContainerType(StrEnum):
    """What role a container plays in the cluster."""

    CENTRAL = "central"
    EXPERT = "expert"


class Config(BaseModel):
    """The cluster layout: which container is central and which experts each other holds.

    Read from config.json on the host, and passed to every node as the CONFIG env var.
    """

    central: ContainerName
    experts: ExpertAlloc

    @field_validator("experts")
    @classmethod
    def _experts_unique(cls, value: ExpertAlloc) -> ExpertAlloc:
        """Each expert id may be assigned to at most one container."""
        seen: set[ExpertId] = set()
        for experts in value.values():
            for expert in experts:
                if expert in seen:
                    raise ValueError(f"expert {expert} is assigned to more than one container")
                seen.add(expert)
        return value

    @model_validator(mode="after")
    def _central_not_expert(self) -> "Config":
        """The central container holds everything except the experts, so it holds none."""
        if self.central in self.experts:
            raise ValueError(f"central container {self.central!r} is also an expert container")
        return self

    def role(self, name: ContainerName) -> ContainerType:
        """What role `name` plays in this cluster."""
        return ContainerType.CENTRAL if name == self.central else ContainerType.EXPERT
