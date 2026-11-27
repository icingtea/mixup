"""Deployment planning: turn a ``LabConfig`` into per-node ``RunSpec``s (pure, docker-free).

This is the policy layer: it decides the image, environment, ports, volumes, GPU mode, and
resource limits for every node — both for the single-host emulated lab (empty endpoints) and
for cross-host deployments (explicit endpoints). The executor (``mixup-lab``) consumes
``RunSpec``s to drive the Docker SDK, and ``mixup-lab render`` prints equivalent
``docker run`` commands. Keeping this module docker-free makes it unit-testable.
"""

import hashlib
import ipaddress
import json
import shlex
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field

from mixup_core.base import StrictModel
from mixup_core.constants import EXPERT_PORT, EXTERNAL_PORT_BASE, INTERNAL_PORT, NETWORK_NAME
from mixup_core.devices import ExecutionType, GpuMode
from mixup_core.env import NODE_ENV_VAR, NodeEnv, dump_node_env
from mixup_core.lab import LabConfig, LabNodeSpec
from mixup_core.types import NodeEndpoint

MODELS_DIR = Path("models")
CONTAINER_MODELS_DIR = "/usr/local/app/models"
LEDGER_DIR = Path("var/ledger")
CONTAINER_LEDGER_DIR = "/usr/local/app/ledger"

_CLUSTER_LABEL = "mixup.cluster"
_NODE_LABEL = "mixup.node"
_HASH_LABEL = "mixup.config_hash"


class PortMapping(StrictModel):
    """One published port: container port -> host port (optionally bound to an IP)."""

    container_port: int
    host_port: int
    bind_ip: str | None = None


class VolumeMount(StrictModel):
    """One mount: a host bind path or a named volume at ``container_path``."""

    container_path: str
    source: str
    kind: Literal["bind", "volume"] = "bind"
    mode: Literal["ro", "rw"] = "rw"


class RunSpec(StrictModel):
    """Everything the executor needs to run one node container."""

    node: str
    docker_host: str = "local"
    image: str
    container_name: str
    environment: dict[str, str]
    ports: list[PortMapping] = []
    volumes: list[VolumeMount] = []
    gpu: Annotated[GpuMode, Field(strict=False)] = GpuMode.NONE
    nano_cpus: int | None = None
    mem_limit_mb: int | None = None
    restart_policy: str = "unless-stopped"
    network: str | None = None
    labels: dict[str, str] = {}


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def resolve_gpu(spec: LabNodeSpec) -> GpuMode:
    """The GPU attachment for a node: explicit ``gpu_mode``, else ``cdi``/``none`` by type."""
    if spec.gpu_mode is not None:
        return spec.gpu_mode
    return GpuMode.CDI if spec.execution is ExecutionType.GPU else GpuMode.NONE


def _host_port(url: str) -> int | None:
    return urlparse(url).port


def _bind_ip(endpoint: NodeEndpoint, spec: LabNodeSpec) -> str | None:
    """The host IP to bind published ports to (explicit override, else the endpoint IP)."""
    if spec.bind_ip:
        return spec.bind_ip
    host = urlparse(endpoint.http).hostname
    if host and _is_ip_literal(host):
        return host
    return None


def _volume_name(cluster: str, node: str) -> str:
    return f"{cluster}-ledger-{node}"


def _model_volume_name(cluster: str) -> str:
    return f"{cluster}-models"


def run_spec_hash(spec: RunSpec) -> str:
    """A short digest of everything that should trigger a container recreate."""
    payload = spec.model_dump(exclude={"labels"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _build_node_env(lab: LabConfig, node: str, prompt: str, max_new_tokens: int) -> NodeEnv:
    return NodeEnv(
        container_name=node,
        config=lab.startup,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        profiles=lab.profiles,
        profile_mode=lab.profile_mode,
    )


def _ports_for(
    lab: LabConfig, node: str, offset: int, spec: LabNodeSpec
) -> list[PortMapping]:
    """Published ports: endpoint-derived for deployments, local offset for the emulated lab."""
    endpoint = lab.startup.endpoints.get(node)
    if endpoint is not None:
        bind_ip = _bind_ip(endpoint, spec)
        control = _host_port(endpoint.http) or (EXTERNAL_PORT_BASE + offset)
        expert = endpoint.expert_port
        return [
            PortMapping(container_port=INTERNAL_PORT, host_port=control, bind_ip=bind_ip),
            PortMapping(container_port=EXPERT_PORT, host_port=expert, bind_ip=bind_ip),
        ]
    return [
        PortMapping(container_port=INTERNAL_PORT, host_port=EXTERNAL_PORT_BASE + offset),
    ]


def _volumes_for(lab: LabConfig, node: str, spec: LabNodeSpec) -> list[VolumeMount]:
    """Local nodes bind the existing ``models/``/``var/ledger``; remote nodes use volumes."""
    if spec.docker_host == "local":
        return [
            VolumeMount(
                container_path=CONTAINER_MODELS_DIR,
                source=str(MODELS_DIR.resolve()),
                kind="bind",
                mode="ro",
            ),
            VolumeMount(
                container_path=CONTAINER_LEDGER_DIR,
                source=str(LEDGER_DIR.resolve()),
                kind="bind",
                mode="rw",
            ),
        ]
    return [
        VolumeMount(
            container_path=CONTAINER_MODELS_DIR,
            source=_model_volume_name(lab.cluster_name),
            kind="volume",
            mode="ro",
        ),
        VolumeMount(
            container_path=CONTAINER_LEDGER_DIR,
            source=_volume_name(lab.cluster_name, node),
            kind="volume",
            mode="rw",
        ),
    ]


def plan_run_specs(
    lab: LabConfig, prompt: str, max_new_tokens: int
) -> dict[str, RunSpec]:
    """Compute the full run plan for every node in the roster."""
    specs: dict[str, RunSpec] = {}
    leader = lab.startup.initial_leader
    others = [n for n in lab.startup.nodes if n != leader]
    for offset, node in enumerate([leader, *others]):
        node_spec = lab.spec_for(node)
        gpu = resolve_gpu(node_spec)
        node_env = _build_node_env(lab, node, prompt, max_new_tokens)
        environment = {NODE_ENV_VAR: dump_node_env(node_env)}
        if gpu is GpuMode.NONE:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        run = RunSpec(
            node=node,
            docker_host=node_spec.docker_host,
            image=lab.image,
            container_name=node,
            environment=environment,
            ports=_ports_for(lab, node, offset, node_spec),
            volumes=_volumes_for(lab, node, node_spec),
            gpu=gpu,
            nano_cpus=int(node_spec.cpus * 1e9) if node_spec.cpus else None,
            mem_limit_mb=node_spec.memory_mb,
            network=None if lab.startup.endpoints else NETWORK_NAME,
            labels={
                _CLUSTER_LABEL: lab.cluster_name,
                _NODE_LABEL: node,
            },
        )
        run.labels[_HASH_LABEL] = run_spec_hash(run)
        specs[node] = run
    return specs


def _gpu_flags(gpu: GpuMode) -> list[str]:
    if gpu is GpuMode.CDI:
        return ["--device", "nvidia.com/gpu=all"]
    if gpu is GpuMode.NVIDIA:
        return ["--gpus", "all"]
    return []


def render_command(spec: RunSpec) -> str:
    """Render one RunSpec as a shell ``docker run`` command (for ``mixup-lab render``)."""
    argv: list[str] = ["docker"]
    if spec.docker_host != "local":
        argv += ["-H", spec.docker_host]
    argv += ["run", "-d", "--name", spec.container_name, "--restart", spec.restart_policy]
    if spec.network is not None:
        argv += ["--network", spec.network]
    if spec.nano_cpus is not None:
        argv += ["--cpus", f"{spec.nano_cpus / 1e9:g}"]
    if spec.mem_limit_mb is not None:
        argv += ["--memory", f"{spec.mem_limit_mb}m"]
    argv += _gpu_flags(spec.gpu)
    for port in spec.ports:
        host = f"{port.bind_ip}:" if port.bind_ip else ""
        argv += ["-p", f"{host}{port.host_port}:{port.container_port}"]
    for volume in spec.volumes:
        argv += ["-v", f"{volume.source}:{volume.container_path}:{volume.mode}"]
    for key, value in spec.environment.items():
        argv += ["-e", f"{key}={value}"]
    argv += [spec.image]
    return shlex.join(argv)
