"""Tests for the cluster Config model and its loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mixup_core.config import load_config_file
from mixup_core.types import Config, ContainerType


def test_valid_config_roles() -> None:
    config = Config(central="container_1", experts={"container_2": [0, 1], "container_3": [2, 3]})
    assert config.role("container_1") is ContainerType.CENTRAL
    assert config.role("container_2") is ContainerType.EXPERT


def test_duplicate_expert_rejected() -> None:
    with pytest.raises(ValidationError):
        Config(central="c1", experts={"c2": [0, 1], "c3": [1, 2]})


def test_central_cannot_hold_experts() -> None:
    with pytest.raises(ValidationError):
        Config(central="c1", experts={"c1": [0, 1], "c2": [2, 3]})


def test_load_config_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"central": "c1", "experts": {"c2": [0, 1]}}')
    config = load_config_file(path)
    assert config.central == "c1"
    assert config.experts == {"c2": [0, 1]}
