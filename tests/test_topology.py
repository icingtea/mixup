"""Tests for expert routing (LUT) and partition validation."""

import pytest

from mixup_core.topology import build_lut, experts_for, validate_partition
from mixup_core.types import Config


def _config() -> Config:
    return Config(
        central="container_1",
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


def test_validate_partition_ok() -> None:
    validate_partition(_config(), num_experts=8)


def test_validate_partition_missing() -> None:
    config = Config(central="c1", experts={"c2": [0, 1], "c3": [4, 5]})
    with pytest.raises(ValueError, match="does not cover"):
        validate_partition(config, num_experts=8)


def test_validate_partition_non_contiguous() -> None:
    config = Config(central="c1", experts={"c2": [0, 1, 3, 4], "c3": [2, 5, 6, 7]})
    with pytest.raises(ValueError, match="not contiguous"):
        validate_partition(config, num_experts=8)
