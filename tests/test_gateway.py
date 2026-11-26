"""Tests for the gateway's leader discovery and failover retry (torch-free)."""

import httpx
from fastapi.testclient import TestClient

from mixup_core.types import GenerateRequest, StartupConfig
from mixup_gateway.gateway import Gateway, build_app


def _config() -> StartupConfig:
    return StartupConfig(
        initial_leader="container_1",
        nodes=["container_1", "container_2", "container_3"],
        experts={"container_2": [0, 1], "container_3": [2, 3]},
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def test_discover_leader_picks_highest_epoch(monkeypatch) -> None:
    projections = {
        "http://localhost:9000/leader": {"leader": "container_1", "epoch": 1},
        "http://localhost:9001/leader": {"leader": "container_2", "epoch": 3},
        "http://localhost:9002/leader": {"leader": "container_3", "epoch": 2},
    }

    def fake_get(url: str, timeout: float) -> _FakeResponse:
        return _FakeResponse(projections[url])

    monkeypatch.setattr(httpx, "get", fake_get)
    assert Gateway(_config()).discover_leader() == "container_2"


def test_generate_retries_after_failure(monkeypatch) -> None:
    gateway = Gateway(_config())
    monkeypatch.setattr(gateway, "discover_leader", lambda: "container_1")
    monkeypatch.setattr(gateway, "_wait_ready", lambda url: True)
    monkeypatch.setattr("mixup_gateway.gateway.time.sleep", lambda s: None)

    calls = {"n": 0}

    def fake_post(url: str, json: dict, timeout: float | None) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("leader down")
        return _FakeResponse(
            {
                "text": "hello",
                "tokens": 1,
                "elapsed_seconds": 1.0,
                "dense_seconds": 0.5,
                "dispatch_seconds": 0.5,
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = gateway.generate(GenerateRequest(prompt="hi"), retries=2)
    assert result is not None
    assert result.text == "hello"
    assert calls["n"] == 2


def test_generate_returns_none_when_no_leader(monkeypatch) -> None:
    gateway = Gateway(_config())
    monkeypatch.setattr(gateway, "discover_leader", lambda: None)
    monkeypatch.setattr("mixup_gateway.gateway.time.sleep", lambda s: None)
    assert gateway.generate(GenerateRequest(prompt="hi"), retries=2) is None


def test_build_app_returns_503_when_no_leader(monkeypatch) -> None:
    gateway = Gateway(_config())
    monkeypatch.setattr(gateway, "discover_leader", lambda: None)
    monkeypatch.setattr("mixup_gateway.gateway.time.sleep", lambda s: None)
    client = TestClient(build_app(gateway))
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
    assert response.status_code == 503
