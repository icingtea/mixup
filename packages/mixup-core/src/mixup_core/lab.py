"""The single-machine heterogeneous lab: config schema for an emulated device pool.

``LabConfig`` wraps a startup layout with per-node ``DeviceProfile``s (what the placement
engine perceives) and per-node ``LabNodeSpec``s (how the lab orchestrator emulates the
device: execution type, CPU quota, memory limit). It is the config the ``mixup-lab``
command consumes to stand up 1 GPU leader + N CPU/GPU experts on this machine.
"""

from typing import Annotated

from pydantic import Field

from .base import StrictModel
from .devices import DeviceProfile, ExecutionType, GpuMode, ProfileMode
from .types import ContainerName, StartupConfig


class LabNodeSpec(StrictModel):
    """How to run one node (emulated on this host, or deployed to a device)."""

    execution: Annotated[ExecutionType, Field(strict=False)] = ExecutionType.GPU
    cpus: float | None = None
    """CPU quota (Docker ``--cpus``); ``None`` = unrestricted."""

    memory_mb: int | None = None
    """Memory limit (Docker ``--memory``); ``None`` = unrestricted."""

    docker_host: str = "local"
    """Where this node's container runs. ``"local"`` = this machine's Docker daemon;
    otherwise a remote daemon reachable via ``ssh://user@host`` or ``tcp://host:port``."""

    gpu_mode: Annotated[GpuMode, Field(strict=False)] | None = None
    """How to attach the GPU (``cdi`` / ``nvidia`` / ``none``). ``None`` = auto: ``cdi``
    for GPU nodes, ``none`` for CPU nodes."""

    bind_ip: str | None = None
    """Host IP to bind the published ports to. ``None`` = auto (the node's endpoint IP
    when it is a literal, else all interfaces)."""

    model_bind: str | None = None
    """Remote host path to bind-mount as the model dir (instead of a named volume + fetch).
    Use this when the model is already on the device (e.g. rsynced to ``~/mixup/models``),
    which avoids a duplicate copy and the HF download."""


class LabConfig(StrictModel):
    """A cluster: startup layout + profiles + per-node run specs."""

    startup: StartupConfig
    profiles: dict[ContainerName, DeviceProfile] = {}
    specs: dict[ContainerName, LabNodeSpec] = {}
    profile_mode: Annotated[ProfileMode, Field(strict=False)] = ProfileMode.HYBRID
    image: str = "mixup:latest"
    """Container image to run (a registry ref for cross-host deployments)."""

    cluster_name: str = "mixup"
    """Label/volume prefix so multiple clusters can coexist on one daemon."""

    def spec_for(self, node: ContainerName) -> LabNodeSpec:
        return self.specs.get(node, LabNodeSpec())

    def profile_for(self, node: ContainerName) -> DeviceProfile | None:
        return self.profiles.get(node)
