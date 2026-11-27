"""Tests for cross-host endpoint resolution (NodeEndpoint, EndpointResolver, node_urls)."""

import pytest

from mixup_core.endpoints import EndpointResolver
from mixup_core.hosts import node_urls
from mixup_core.types import NodeEndpoint, StartupConfig


def _config() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2"],
        experts={"container_2": [0, 1]},
    )


def test_node_endpoint_requires_http_scheme() -> None:
    with pytest.raises(ValueError):
        NodeEndpoint(http="tcp://host:9000", expert_host="host")


def test_node_endpoint_requires_expert_host() -> None:
    with pytest.raises(ValueError):
        NodeEndpoint(http="http://host:9000", expert_host="  ")


def test_endpoints_must_cover_roster() -> None:
    with pytest.raises(ValueError, match="endpoints must match nodes exactly"):
        StartupConfig(
            initial_leader="container_1",
            nodes=["container_1", "container_2"],
            experts={"container_2": [0, 1]},
            endpoints={"container_1": NodeEndpoint(http="http://a:9000", expert_host="a")},
        )
    with pytest.raises(ValueError, match="endpoints must match nodes exactly"):
        StartupConfig(
            initial_leader="container_1",
            nodes=["container_1"],
            experts={},
            endpoints={
                "container_1": NodeEndpoint(http="http://a:9000", expert_host="a"),
                "container_2": NodeEndpoint(http="http://b:9000", expert_host="b"),
            },
        )


def test_resolver_falls_back_to_docker_defaults() -> None:
    resolver = EndpointResolver(_config())
    assert resolver.http("container_1") == "http://container_1:8000"
    assert resolver.expert("container_2") == ("container_2", 8001)


def test_resolver_uses_explicit_endpoints() -> None:
    config = StartupConfig(
        initial_leader="pc",
        nodes=["pc", "laptop"],
        experts={"laptop": [0, 1]},
        endpoints={
            "pc": NodeEndpoint(
                http="http://100.102.170.72:9000", expert_host="100.102.170.72", expert_port=9001
            ),
            "laptop": NodeEndpoint(
                http="http://100.90.0.1:9000", expert_host="100.90.0.1", expert_port=9001
            ),
        },
    )
    resolver = EndpointResolver(config)
    assert resolver.http("pc") == "http://100.102.170.72:9000"
    assert resolver.expert("pc") == ("100.102.170.72", 9001)
    assert resolver.expert("laptop") == ("100.90.0.1", 9001)


def test_node_urls_prefers_endpoints() -> None:
    config = StartupConfig(
        initial_leader="pc",
        nodes=["pc", "laptop"],
        experts={"laptop": [0, 1]},
        endpoints={
            "pc": NodeEndpoint(http="http://100.102.170.72:9000", expert_host="100.102.170.72"),
            "laptop": NodeEndpoint(http="http://100.90.0.1:9000", expert_host="100.90.0.1"),
        },
    )
    assert node_urls(config) == {
        "pc": "http://100.102.170.72:9000",
        "laptop": "http://100.90.0.1:9000",
    }


def test_node_urls_falls_back_to_local_ports() -> None:
    assert node_urls(_config()) == {
        "container_1": "http://localhost:9000",
        "container_2": "http://localhost:9001",
    }
