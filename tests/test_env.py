"""Tests for the node runtime environment (NodeEnv)."""

import pytest

from mixup_core.env import NODE_ENV_VAR, NodeEnv, dump_node_env, load_node_env
from mixup_core.types import Config


def test_node_env_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    env = NodeEnv(
        container_name="container_2",
        config=Config(central="container_1", experts={"container_2": [0, 1]}),
        prompt="hello world",
        max_new_tokens=7,
    )
    monkeypatch.setenv(NODE_ENV_VAR, dump_node_env(env))
    loaded = load_node_env()
    assert loaded == env
    assert loaded.container_name == "container_2"
    assert loaded.config.central == "container_1"
    assert loaded.prompt == "hello world"
    assert loaded.max_new_tokens == 7


def test_node_env_defaults() -> None:
    env = NodeEnv(
        container_name="container_1",
        config=Config(central="container_1", experts={"container_2": [0, 1]}),
    )
    assert env.prompt == "What is mixture of experts?"
    assert env.max_new_tokens == 50
