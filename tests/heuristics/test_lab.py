"""Tests for the lab config schema."""

from mixup_core.devices import DeviceProfile, ExecutionType, ProfileMode
from mixup_core.lab import LabConfig, LabNodeSpec
from mixup_core.types import StartupConfig


def _startup() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2", "container_3"],
        experts={"container_2": [0, 1], "container_3": [2, 3]},
    )


def test_lab_config_defaults() -> None:
    lab = LabConfig(startup=_startup())
    assert lab.profile_mode is ProfileMode.HYBRID
    assert lab.spec_for("container_2").execution is ExecutionType.GPU  # default
    assert lab.profile_for("container_2") is None


def test_lab_config_spec_and_profile() -> None:
    lab = LabConfig(
        startup=_startup(),
        profiles={"container_3": DeviceProfile(node_id="container_3", execution="cpu")},
        specs={"container_3": LabNodeSpec(execution="cpu", cpus=2, memory_mb=6000)},
    )
    spec = lab.spec_for("container_3")
    assert spec.execution is ExecutionType.CPU
    assert spec.cpus == 2
    assert lab.profile_for("container_3").is_gpu is False


def test_lab_config_leader_may_own_experts() -> None:
    startup = StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2"],
        experts={"container_1": [0, 1], "container_2": [2, 3]},
    )
    lab = LabConfig(startup=startup)
    assert "container_1" in lab.startup.experts
