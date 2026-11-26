"""Tests for the REPL's discovery logic (torch-free)."""

from mixup_core.hosts import node_urls
from mixup_core.types import StartupConfig
from mixup_repl.repl import pick_leader


def _config() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2", "container_3"],
        experts={"container_2": [0, 1], "container_3": [2, 3]},
    )


def test_node_urls_mirrors_orchestrator_ports() -> None:
    urls = node_urls(_config())
    assert urls == {
        "container_1": "http://localhost:9000",
        "container_2": "http://localhost:9001",
        "container_3": "http://localhost:9002",
    }


def test_pick_leader_prefers_highest_epoch() -> None:
    projections = [
        {"leader": "container_1", "epoch": 1},
        {"leader": "container_2", "epoch": 3},
        {"leader": "container_3", "epoch": 2},
    ]
    assert pick_leader(projections) == "container_2"


def test_pick_leader_ignores_none() -> None:
    assert pick_leader([{"leader": None, "epoch": 5}]) is None
    assert pick_leader([]) is None
