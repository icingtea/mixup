"""Tests for the gossip loop (the background driver of the SWIM protocol)."""

from mixup_core.gossip_engine import GossipEngine
from mixup_core.runtime import ClusterView, MemberState
from mixup_core.types import NodeRole
from mixup_node.gossip_loop import GossipLoop


def _member(node: str) -> MemberState:
    return MemberState(node_id=node, role=NodeRole.EXPERT, epoch=0, incarnation=1, priority=1)


def _engine() -> GossipEngine:
    view = ClusterView(
        members={
            "a": _member("a"),
            "b": _member("b"),
            "c": _member("c"),
            "d": _member("d"),
        }
    )
    return GossipEngine(
        node_id="a", view=view, ping_timeout=1.0, suspect_timeout=3.0, indirect_k=1
    )


class QueueChooser:
    """Return peers from a queue, then None. Ignores the exclude set for determinism."""

    def __init__(self, *peers: str) -> None:
        self._it = iter(peers)

    def __call__(self, exclude: frozenset[str] = frozenset()) -> str | None:
        return next(self._it, None)


def _loop(engine: GossipEngine, *peers: str, indirect_k: int = 1) -> tuple[GossipLoop, list]:
    sent: list = []

    def send(dest, msg) -> None:
        sent.append((dest, msg))

    loop = GossipLoop(
        engine,
        gossip_interval=1.0,
        ping_timeout=1.0,
        suspect_timeout=3.0,
        indirect_k=indirect_k,
        send=send,
        choose_peer=QueueChooser(*peers),
    )
    return loop, sent


def test_tick_sends_initial_ping() -> None:
    loop, sent = _loop(_engine(), "b")
    loop.tick(0.0)

    assert len(sent) == 1
    dest, msg = sent[0]
    assert dest == "b"
    assert msg["type"] == "ping"


def test_no_ping_before_gossip_interval() -> None:
    loop, sent = _loop(_engine(), "b")
    loop.tick(0.0)  # pings b
    loop.tick(0.5)  # before gossip_interval elapses again
    assert len(sent) == 1  # still just the one ping


def test_ping_timeout_sends_indirect_probe() -> None:
    loop, sent = _loop(_engine(), "b", "c")
    loop.tick(0.0)   # ping b
    loop.tick(1.0)   # b times out -> PingReq

    pingreq = [m for dest, m in sent if m["type"] == "pingreq"]
    assert len(pingreq) == 1
    assert pingreq[0]["target"] == "b"


def test_indirect_timeout_marks_suspect() -> None:
    loop, sent = _loop(_engine(), "b", "c")
    loop.tick(0.0)   # ping b
    loop.tick(1.0)   # PingReq
    loop.tick(2.0)   # indirect failed -> Suspect

    suspect = [m for dest, m in sent if m["type"] == "suspect"]
    assert len(suspect) == 1
    # broadcast: destination is None
    assert any(dest is None and m["type"] == "suspect" for dest, m in sent)


def test_suspect_timeout_marks_dead() -> None:
    loop, sent = _loop(_engine(), "b", "c")
    loop.tick(0.0)   # ping b
    loop.tick(1.0)   # PingReq
    loop.tick(2.0)   # Suspect (suspect since 2.0)
    loop.tick(5.0)   # suspect_timeout elapsed -> Dead

    dead = [m for dest, m in sent if m["type"] == "dead"]
    assert len(dead) == 1
    assert dead[0]["node_id"] == "b"
    assert any(dest is None and m["type"] == "dead" for dest, m in sent)
