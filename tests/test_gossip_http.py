"""Tests for the gossip HTTP transport (torch-free FastAPI endpoints)."""

from fastapi.testclient import TestClient

from mixup_core.gossip import Ping, encode_gossip
from mixup_core.runtime import ClusterView, MemberState, RoutingEntry, RoutingTable
from mixup_core.types import NodeRole
from mixup_node.gossip_http import GossipServer, create_app


def _member(node: str, priority: int = 1) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        epoch=0,
        incarnation=1,
        priority=priority,
    )


def _server() -> GossipServer:
    view = ClusterView(
        members={
            "a": _member("a"),
            "b": _member("b"),
            "c": _member("c"),
        }
    )
    return GossipServer(
        node_id="a",
        view=view,
        ping_timeout=1.0,
        suspect_timeout=3.0,
        indirect_k=2,
        clock=lambda: 0.0,
    )


def test_health() -> None:
    client = TestClient(create_app(_server()))
    assert client.get("/health").json() == {"status": "ok"}


def test_gossip_ping_returns_ack() -> None:
    client = TestClient(create_app(_server()))
    body = encode_gossip(Ping(node_id="b", incarnation=1, epoch=0, kv_seq=0))
    resp = client.post("/gossip", json=body).json()

    replies = resp["replies"]
    assert len(replies) == 1
    assert replies[0]["to"] == "b"
    assert replies[0]["msg"]["type"] == "ack"
    assert replies[0]["msg"]["node_id"] == "a"


def test_leader_endpoint() -> None:
    client = TestClient(create_app(_server()))
    proj = client.get("/leader").json()
    assert set(proj) == {"leader", "epoch", "kv_seq", "routing_epoch"}
    assert proj["leader"] in {"a", "b", "c"}  # deterministic winner among equals


def test_routing_roundtrip() -> None:
    server = _server()
    client = TestClient(create_app(server))

    assert client.get("/routing").json() == {}

    table = RoutingTable(epoch=1, assignment={0: RoutingEntry(node_id="b", slot=0)})
    assert client.post("/routing", json=table.model_dump()).json() == {"ok": True}
    got = client.get("/routing").json()
    assert got["epoch"] == 1
    assert got["assignment"]["0"]["node_id"] == "b"


def test_routing_ignores_older_table() -> None:
    server = _server()
    client = TestClient(create_app(server))

    new = RoutingTable(epoch=2, assignment={})
    old = RoutingTable(epoch=1, assignment={})
    client.post("/routing", json=new.model_dump())
    client.post("/routing", json=old.model_dump())
    assert client.get("/routing").json()["epoch"] == 2
