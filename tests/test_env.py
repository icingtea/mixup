"""Tests for the node runtime environment (NodeEnv)."""

import pytest

from mixup_core.env import NODE_ENV_VAR, NodeEnv, dump_node_env, load_node_env
from mixup_core.types import StartupConfig


def _config() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2"],
        experts={"container_2": [0, 1]},
    )


def test_node_env_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    env = NodeEnv(
        container_name="container_2",
        config=_config(),
        prompt="hello world",
        max_new_tokens=7,
    )
    monkeypatch.setenv(NODE_ENV_VAR, dump_node_env(env))
    loaded = load_node_env()
    assert loaded == env
    assert loaded.container_name == "container_2"
    assert loaded.config.initial_leader == "container_1"
    assert loaded.prompt == "hello world"
    assert loaded.max_new_tokens == 7


def test_node_env_defaults() -> None:
    env = NodeEnv(container_name="container_1", config=_config())
    assert env.prompt == "What is mixture of experts?"
    assert env.max_new_tokens == 50
