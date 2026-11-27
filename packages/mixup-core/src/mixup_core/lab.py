"""The single-machine heterogeneous lab: config schema for an emulated device pool.

``LabConfig`` wraps a startup layout with per-node ``DeviceProfile``s (what the placement
engine perceives) and per-node ``LabNodeSpec``s (how the lab orchestrator emulates the
device: execution type, CPU quota, memory limit). It is the config the ``mixup-lab``
command consumes to stand up 1 GPU leader + N CPU/GPU experts on this machine.
"""

from typing import Annotated

from pydantic import Field

from .base import StrictModel
from .devices import DeviceProfile, ExecutionType, ProfileMode
from .types import ContainerName, StartupConfig


class LabNodeSpec(StrictModel):
    """How to emulate one node in the lab."""

    execution: Annotated[ExecutionType, Field(strict=False)] = ExecutionType.GPU
    cpus: float | None = None
    """CPU quota (Docker ``--cpus``); ``None`` = unrestricted."""

    memory_mb: int | None = None
    """Memory limit (Docker ``--memory``); ``None`` = unrestricted."""


class LabConfig(StrictModel):
    """A lab cluster: startup layout + profiles + emulation specs."""

    startup: StartupConfig
    profiles: dict[ContainerName, DeviceProfile] = {}
    specs: dict[ContainerName, LabNodeSpec] = {}
    profile_mode: Annotated[ProfileMode, Field(strict=False)] = ProfileMode.HYBRID

    def spec_for(self, node: ContainerName) -> LabNodeSpec:
        return self.specs.get(node, LabNodeSpec())

    def profile_for(self, node: ContainerName) -> DeviceProfile | None:
        return self.profiles.get(node)
