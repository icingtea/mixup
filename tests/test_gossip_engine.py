"""Tests for the gossip protocol engine (message handling + failure detection + relay)."""

from mixup_core.failure_detection import Action
from mixup_core.gossip import Ack, Alive, Dead, Ping, PingReq, Suspect
from mixup_core.gossip_engine import GossipEngine
from mixup_core.runtime import ClusterView, MemberState
from mixup_core.types import MemberStatus, NodeRole


def _member(
    node: str,
    incarnation: int = 1,
    status: MemberStatus = MemberStatus.ALIVE,
) -> MemberState:
    return MemberState(
        node_id=node,
        role=NodeRole.EXPERT,
        status=status,
        epoch=0,
        incarnation=incarnation,
        priority=1,
    )


def _engine() -> GossipEngine:
    view = ClusterView(
        members={
            "a": _member("a"),
            "b": _member("b"),
            "c": _member("c"),
        }
    )
    return GossipEngine(
        node_id="a",
        view=view,
        ping_timeout=1.0,
        suspect_timeout=3.0,
        indirect_k=2,
    )


def test_ping_replies_with_ack_to_sender() -> None:
    out = _engine().handle(Ping(node_id="b", incarnation=1, epoch=0, kv_seq=0), now=0.0)
    assert len(out) == 1
    dest, msg = out[0]
    assert dest == "b"
    assert isinstance(msg, Ack)
    assert msg.node_id == "a"  # ack for self


def test_ack_without_relay_or_revival_is_quiet() -> None:
    out = _engine().handle(Ack(node_id="b", incarnation=1, epoch=0, kv_seq=0), now=0.0)
    assert out == []


def test_ack_after_suspect_broadcasts_alive() -> None:
    engine = _engine()
    engine.probe("b", now=0.0)
    engine.on_timeout("b", now=1.0)  # PINGREQ
    action, _ = engine.on_timeout("b", now=2.0)  # SUSPECT
    assert action is Action.SUSPECT

    out = engine.handle(Ack(node_id="b", incarnation=1, epoch=0, kv_seq=0), now=2.5)
    assert (None, out[0][1]) == out[0] and isinstance(out[0][1], Alive)
    assert out[0][1].node_id == "b"


def test_pingreq_triggers_relay_on_ack() -> None:
    engine = _engine()
    out = engine.handle(PingReq(from_node="b", target="c", incarnation=1), now=0.0)
    assert len(out) == 1
    dest, msg = out[0]
    assert dest == "c"
    assert isinstance(msg, Ping)

    # c acks our indirect probe -> we relay an Ack back to the requester b.
    out = engine.handle(Ack(node_id="c", incarnation=1, epoch=0, kv_seq=0), now=0.5)
    assert len(out) == 1
    dest, msg = out[0]
    assert dest == "b"
    assert isinstance(msg, Ack)
    assert msg.node_id == "c"


def test_suspect_message_updates_view() -> None:
    engine = _engine()
    assert engine.handle(Suspect(node_id="b", incarnation=1, by_node="c"), now=0.0) == []
    assert engine.view.members["b"].status is MemberStatus.SUSPECT


def test_dead_message_updates_view() -> None:
    engine = _engine()
    engine.handle(Dead(node_id="b", incarnation=1, by_node="c"), now=0.0)
    assert engine.view.members["b"].status is MemberStatus.DEAD


def test_on_timeout_suspect_returns_message() -> None:
    engine = _engine()
    engine.probe("b", now=0.0)
    engine.on_timeout("b", now=1.0)  # PINGREQ
    action, msg = engine.on_timeout("b", now=2.0)
    assert action is Action.SUSPECT
    assert isinstance(msg, Suspect)
    assert msg.node_id == "b"
    assert engine.view.members["b"].status is MemberStatus.SUSPECT


def test_on_timeout_dead_returns_message() -> None:
    engine = _engine()
    engine.probe("b", now=0.0)
    engine.on_timeout("b", now=1.0)  # PINGREQ
    engine.on_timeout("b", now=2.0)  # SUSPECT
    action, msg = engine.on_timeout("b", now=5.0)  # suspect_timeout elapsed
    assert action is Action.DEAD
    assert isinstance(msg, Dead)
    assert msg.node_id == "b"


def test_on_timeout_pingreq_has_no_broadcast() -> None:
    engine = _engine()
    engine.probe("b", now=0.0)
    action, msg = engine.on_timeout("b", now=1.0)
    assert action is Action.PINGREQ
    assert msg is None


def test_mutate_applies_fn_and_sticks() -> None:
    engine = _engine()
    assert engine.view.members["b"].status is MemberStatus.ALIVE
    new_view = engine.mutate(
        lambda v: ClusterView(
            members={k: m for k, m in v.members.items() if k != "b"}, routing=None
        )
    )
    assert "b" not in new_view.members
    assert "b" not in engine.view.members  # the mutation persisted in the shared view


def test_ping_merges_peer_kv_seq_and_epoch() -> None:
    engine = _engine()
    engine.handle(Ping(node_id="b", incarnation=1, epoch=2, kv_seq=42), now=0.0)
    assert engine.view.members["b"].epoch == 2
    assert engine.view.members["b"].kv_seq == 42


def test_ack_merges_peer_kv_seq() -> None:
    engine = _engine()
    engine.handle(Ack(node_id="c", incarnation=1, epoch=0, kv_seq=7), now=0.0)
    assert engine.view.members["c"].kv_seq == 7


def test_stale_incarnation_is_not_merged() -> None:
    engine = _engine()
    engine.handle(Ping(node_id="b", incarnation=0, epoch=5, kv_seq=99), now=0.0)
    assert engine.view.members["b"].epoch == 0
    assert engine.view.members["b"].kv_seq == 0


def test_higher_incarnation_revives_dead_peer() -> None:
    engine = _engine()
    engine.handle(Dead(node_id="b", incarnation=1, by_node="c"), now=0.0)
    assert engine.view.members["b"].status is MemberStatus.DEAD
    # b restarts with a fresh (higher) incarnation -> its Ping revives it.
    engine.handle(Ping(node_id="b", incarnation=2, epoch=0, kv_seq=0), now=0.5)
    assert engine.view.members["b"].status is MemberStatus.ALIVE
    assert engine.view.members["b"].incarnation == 2


def test_same_incarnation_does_not_revive_dead_peer() -> None:
    engine = _engine()
    engine.handle(Dead(node_id="b", incarnation=1, by_node="c"), now=0.0)
    engine.handle(Ping(node_id="b", incarnation=1, epoch=0, kv_seq=0), now=0.5)
    assert engine.view.members["b"].status is MemberStatus.DEAD


def test_adopt_routing_adopts_newer_and_ignores_stale() -> None:
    from mixup_core.runtime import RoutingEntry, RoutingTable

    engine = _engine()
    table = RoutingTable(
        epoch=3, assignment={0: RoutingEntry(node_id="b", slot=0)}
    )
    assert engine.adopt_routing(table) is True
    assert engine.view.routing is not None and engine.view.routing.epoch == 3

    stale = RoutingTable(epoch=2, assignment={0: RoutingEntry(node_id="c", slot=0)})
    assert engine.adopt_routing(stale) is False
    assert engine.view.routing.epoch == 3  # unchanged


def test_adopt_routing_runs_callback_under_lock() -> None:
    from mixup_core.runtime import RoutingEntry, RoutingTable

    engine = _engine()
    seen: list[int] = []
    table = RoutingTable(epoch=1, assignment={0: RoutingEntry(node_id="b", slot=0)})
    assert engine.adopt_routing(table, on_adopted=lambda t: seen.append(t.epoch)) is True
    assert seen == [1]
