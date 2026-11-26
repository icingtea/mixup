"""Tests for expert routing (LUT) and partition validation."""

import pytest

from mixup_core.topology import build_lut, expert_containers, experts_for, validate_partition
from mixup_core.types import StartupConfig


def _config() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2", "container_3"],
        experts={
            "container_2": [0, 1, 2, 3],
            "container_3": [4, 5, 6, 7],
        },
    )


def test_build_lut_routes_by_explicit_lists() -> None:
    lut = build_lut(_config())
    assert lut[0] == ("container_2", 0)
    assert lut[3] == ("container_2", 3)
    assert lut[4] == ("container_3", 0)
    assert lut[7] == ("container_3", 3)


def test_experts_for() -> None:
    lut = build_lut(_config())
    assert experts_for(lut, "container_2") == [0, 1, 2, 3]
    assert experts_for(lut, "container_3") == [4, 5, 6, 7]


def test_expert_containers_in_config_order() -> None:
    assert expert_containers(_config()) == ["container_2", "container_3"]


def test_build_lut_slot_is_list_position_not_value() -> None:
    config = StartupConfig(
        initial_leader="c1",
        nodes=["c1", "c2"],
        experts={"c2": [7, 3, 9]},  # unsorted, non-contiguous
    )
    lut = build_lut(config)
    assert lut[7] == ("c2", 0)
    assert lut[3] == ("c2", 1)
    assert lut[9] == ("c2", 2)


def test_experts_for_returns_sorted() -> None:
    config = StartupConfig(
        initial_leader="c1",
        nodes=["c1", "c2"],
        experts={"c2": [9, 3, 7]},
    )
    lut = build_lut(config)
    assert experts_for(lut, "c2") == [3, 7, 9]


def test_validate_partition_ok() -> None:
    validate_partition(_config(), num_experts=8)


def test_validate_partition_missing() -> None:
    config = StartupConfig(
        initial_leader="c1",
        nodes=["c1", "c2", "c3"],
        experts={"c2": [0, 1], "c3": [4, 5]},
    )
    with pytest.raises(ValueError, match="does not cover"):
        validate_partition(config, num_experts=8)


def test_validate_partition_non_contiguous() -> None:
    config = StartupConfig(
        initial_leader="c1",
        nodes=["c1", "c2", "c3"],
        experts={"c2": [0, 1, 3, 4], "c3": [2, 5, 6, 7]},
    )
    with pytest.raises(ValueError, match="not contiguous"):
        validate_partition(config, num_experts=8)
