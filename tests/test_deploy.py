"""Tests for the pure deployment planner (RunSpec / plan_run_specs / render)."""

import shlex

from mixup_core.constants import NETWORK_NAME
from mixup_core.devices import ExecutionType, GpuMode
from mixup_core.lab import LabConfig, LabNodeSpec
from mixup_core.types import NodeEndpoint, StartupConfig
from mixup_orchestrator.deploy import (
    CONTAINER_MODELS_DIR,
    plan_run_specs,
    render_command,
    resolve_gpu,
    run_spec_hash,
)


def _local_lab() -> LabConfig:
    return LabConfig(
        startup=StartupConfig(
            initial_leader="container_1",
            nodes=["container_1", "container_2", "container_3"],
            experts={"container_2": [0, 1], "container_3": [2, 3]},
        ),
        specs={
            "container_1": LabNodeSpec(execution=ExecutionType.GPU),
            "container_2": LabNodeSpec(execution=ExecutionType.GPU),
            "container_3": LabNodeSpec(execution=ExecutionType.CPU, cpus=2.0, memory_mb=6144),
        },
    )


def _deployed() -> LabConfig:
    return LabConfig(
        cluster_name="home",
        image="ghcr.io/icingtea/mixup:latest",
        startup=StartupConfig(
            initial_leader="monad-nix",
            nodes=["monad-nix", "monad-windows", "pi-inch-nails"],
            experts={"monad-windows": [0, 1], "pi-inch-nails": [2, 3]},
            endpoints={
                "monad-nix": NodeEndpoint(
                    http="http://100.102.170.72:9000",
                    expert_host="100.102.170.72",
                    expert_port=9001,
                ),
                "monad-windows": NodeEndpoint(
                    http="http://100.90.0.1:9000", expert_host="100.90.0.1", expert_port=9001
                ),
                "pi-inch-nails": NodeEndpoint(
                    http="http://100.89.94.89:9000", expert_host="100.89.94.89", expert_port=9001
                ),
            },
        ),
        specs={
            "monad-nix": LabNodeSpec(execution=ExecutionType.GPU, docker_host="local"),
            "monad-windows": LabNodeSpec(
                execution=ExecutionType.GPU,
                docker_host="ssh://lait@monad-windows",
                gpu_mode=GpuMode.NVIDIA,
            ),
            "pi-inch-nails": LabNodeSpec(
                execution=ExecutionType.CPU, docker_host="ssh://lait@pi-inch-nails"
            ),
        },
    )


def test_resolve_gpu_modes() -> None:
    assert resolve_gpu(LabNodeSpec(execution=ExecutionType.GPU)) is GpuMode.CDI
    assert resolve_gpu(LabNodeSpec(execution=ExecutionType.CPU)) is GpuMode.NONE
    assert (
        resolve_gpu(LabNodeSpec(execution=ExecutionType.GPU, gpu_mode=GpuMode.NVIDIA))
        is GpuMode.NVIDIA
    )


def test_local_lab_plan() -> None:
    specs = plan_run_specs(_local_lab(), "hi", 10)
    assert set(specs) == {"container_1", "container_2", "container_3"}

    leader = specs["container_1"]
    assert leader.ports[0].host_port == 9000
    assert leader.ports[0].container_port == 8000
    assert leader.network == NETWORK_NAME
    assert leader.volumes[0].kind == "bind"
    assert leader.volumes[0].container_path == CONTAINER_MODELS_DIR
    assert leader.gpu is GpuMode.CDI
    assert "CUDA_VISIBLE_DEVICES" not in leader.environment

    cpu = specs["container_3"]
    assert cpu.ports[0].host_port == 9002
    assert cpu.gpu is GpuMode.NONE
    assert cpu.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu.mem_limit_mb == 6144
    assert cpu.nano_cpus == int(2.0 * 1e9)


def test_deployed_plan_ports_and_volumes() -> None:
    specs = plan_run_specs(_deployed(), "hi", 10)

    remote = specs["monad-windows"]
    assert remote.docker_host == "ssh://lait@monad-windows"
    assert remote.network is None
    assert remote.network_mode == "host"
    assert remote.gpu is GpuMode.NVIDIA
    assert remote.ports == []  # host networking: no port publishing
    assert remote.volumes[0].kind == "volume"
    assert remote.volumes[0].source == "home-models"

    local = specs["monad-nix"]
    assert local.docker_host == "local"
    assert local.volumes[0].kind == "bind"
    assert local.gpu is GpuMode.CDI


def test_run_spec_hash_detects_change() -> None:
    specs = plan_run_specs(_deployed(), "hi", 10)
    a = run_spec_hash(specs["monad-windows"])
    b = plan_run_specs(_deployed(), "hi", 11)
    assert a != run_spec_hash(b["monad-windows"])


def test_render_command_parses() -> None:
    specs = plan_run_specs(_deployed(), "hi", 10)
    cmd = render_command(specs["monad-windows"])
    argv = shlex.split(cmd)
    assert argv[0] == "docker"
    assert "-H" in argv and "ssh://lait@monad-windows" in argv
    assert "--gpus" in argv
    assert specs["monad-windows"].image in argv
    # the env JSON survives a shell parse round-trip
    env_idx = argv.index("-e") + 1
    assert argv[env_idx].startswith("MIXUP_NODE_ENV=")
