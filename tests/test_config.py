"""Tests for the StartupConfig model and its loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mixup_core.startup_config import load_startup_config
from mixup_core.types import NodeRole, StartupConfig


def _config(**overrides: object) -> StartupConfig:
    base: dict[str, object] = {
        "initial_leader": "container_1",
        "nodes": ["container_1", "container_2", "container_3"],
        "experts": {"container_2": [0, 1], "container_3": [2, 3]},
    }
    base.update(overrides)
    return StartupConfig(**base)


def test_valid_config_roles() -> None:
    config = _config()
    assert config.initial_role("container_1") is NodeRole.LEADER
    assert config.initial_role("container_2") is NodeRole.EXPERT


def test_duplicate_expert_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(experts={"container_2": [0, 1], "container_3": [1, 2]})


def test_leader_may_hold_experts() -> None:
    # Leader-as-owner (Phase 1): the initial leader may also serve experts.
    config = _config(experts={"container_1": [0, 1], "container_2": [2, 3]})
    assert config.initial_role("container_1") is NodeRole.LEADER
    assert config.experts["container_1"] == [0, 1]


def test_load_startup_config(tmp_path: Path) -> None:
    path = tmp_path / "startup_config.json"
    path.write_text(
        '{"initial_leader": "c1", "nodes": ["c1", "c2"], "experts": {"c2": [0, 1]}}'
    )
    config = load_startup_config(path)
    assert config.initial_leader == "c1"
    assert config.nodes == ["c1", "c2"]
    assert config.experts == {"c2": [0, 1]}


def test_load_startup_config_from_lab_config(tmp_path: Path) -> None:
    path = tmp_path / "lab_config.json"
    path.write_text(
        '{"startup": {"initial_leader": "c1", "nodes": ["c1", "c2"], '
        '"experts": {"c2": [0, 1]}}, "profiles": {}, "specs": {}}'
    )
    config = load_startup_config(path)
    assert config.initial_leader == "c1"
    assert config.experts == {"c2": [0, 1]}


def test_nodes_parsed() -> None:
    config = _config()
    assert config.nodes == ["container_1", "container_2", "container_3"]


def test_empty_experts_is_valid() -> None:
    config = StartupConfig(initial_leader="c1", nodes=["c1"], experts={})
    assert config.experts == {}
    assert config.initial_role("c1") is NodeRole.LEADER


def test_duplicate_within_same_node_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(experts={"container_2": [0, 0], "container_3": [2, 3]})


def test_duplicate_across_three_nodes_rejected() -> None:
    with pytest.raises(ValidationError):
        StartupConfig(
            initial_leader="c1",
            nodes=["c1", "c2", "c3", "c4"],
            experts={"c2": [0], "c3": [1], "c4": [0]},
        )


def test_initial_role_expert_for_unknown() -> None:
    assert _config().initial_role("container_9") is NodeRole.EXPERT
