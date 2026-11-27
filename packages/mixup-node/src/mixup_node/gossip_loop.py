"""The gossip loop: drives the SWIM protocol on a schedule.

Torch-free, like ``gossip_http``: it owns only the timing (pings, timeouts) and calls an
injected ``send`` to emit outbound messages. ``tick(now)`` takes an explicit time so the
whole loop is unit-testable with synthetic time; ``run()`` is the thin background wrapper
that calls ``tick`` on a real clock.
"""

import logging
import time
from collections.abc import Callable

from mixup_core.failure_detection import Action
from mixup_core.gossip import PingReq, encode_gossip
from mixup_core.gossip_engine import GossipEngine
from mixup_core.types import MemberStatus, NodeId

logger = logging.getLogger(__name__)

Send = Callable[[NodeId | None, dict], None]
ChoosePeer = Callable[[frozenset[NodeId]], NodeId | None]
Clock = Callable[[], float]


class GossipLoop:
    """Pings peers on a schedule, detects probe timeouts, and emits the resulting gossip."""

    def __init__(
        self,
        engine: GossipEngine,
        *,
        gossip_interval: float,
        ping_timeout: float,
        suspect_timeout: float,
        indirect_k: int,
        send: Send | None = None,
        choose_peer: ChoosePeer | None = None,
        on_dead: Callable[[NodeId], None] | None = None,
        clock: Clock = time.monotonic,
        grace_seconds: float = 15.0,
    ) -> None:
        self.engine = engine
        self.gossip_interval = gossip_interval
        self.ping_timeout = ping_timeout
        self.suspect_timeout = suspect_timeout
        self.indirect_k = indirect_k
        self._send = send if send is not None else lambda dest, msg: None
        self._choose_peer = choose_peer if choose_peer is not None else self._default_peer
        self._on_dead = on_dead
        self._clock = clock
        self.grace_seconds = grace_seconds
        self._grace_until = 0.0

        # peer -> next time to advance its probe state
        self._pending: dict[NodeId, float] = {}
        self._next_ping = 0.0

    def _default_peer(self, exclude: frozenset[NodeId] = frozenset()) -> NodeId | None:
        for member in self.engine.view.members.values():
            if (
                member.status == MemberStatus.ALIVE
                and member.node_id != self.engine.node_id
                and member.node_id not in exclude
            ):
                return member.node_id
        return None

    def tick(self, now: float) -> None:
        """Advance the loop one step: handle due timeouts, then maybe send a fresh ping."""
        self._process_timeouts(now)
        if now >= self._next_ping:
            self._next_ping = now + self.gossip_interval
            peer = self._choose_peer(frozenset(self._pending))
            if peer is not None:
                ping = self.engine.probe(peer, now)
                self._pending[peer] = now + self.ping_timeout
                self._send(peer, encode_gossip(ping))

    def _process_timeouts(self, now: float) -> None:
        if now < self._grace_until:
            # During the boot grace period, don't advance SUSPECT/DEAD: a peer still
            # loading its model may be slow to ACK (GIL-heavy torch load), and a spurious
            # DEAD would trigger an unnecessary failover. Just clear pending and re-probe.
            self._pending.clear()
            return
        for peer in [p for p, t in self._pending.items() if now >= t]:
            action, msg = self.engine.on_timeout(peer, now)
            if action is Action.PINGREQ:
                self._send_indirect_probes(peer, now)
                self._pending[peer] = now + self.ping_timeout
            elif action is Action.SUSPECT:
                if msg is not None:
                    self._send(None, encode_gossip(msg))
                self._pending[peer] = now + self.suspect_timeout
                logger.info("gossip: %s is suspect", peer)
            elif action is Action.DEAD:
                if msg is not None:
                    self._send(None, encode_gossip(msg))
                del self._pending[peer]
                logger.info("gossip: %s is dead", peer)
                if self._on_dead is not None:
                    self._on_dead(peer)
            else:  # WAIT (should not normally occur with correct scheduling)
                self._pending[peer] = now + self.suspect_timeout

    def _send_indirect_probes(self, target: NodeId, now: float) -> None:
        exclude = {target}
        for _ in range(self.indirect_k):
            peer = self._choose_peer(frozenset(exclude | set(self._pending)))
            if peer is None:
                break
            exclude.add(peer)
            member = self.engine.view.members.get(target)
            incarnation = member.incarnation if member else 0
            msg = PingReq(from_node=self.engine.node_id, target=target, incarnation=incarnation)
            self._send(peer, encode_gossip(msg))

    def run(self, interval: float) -> None:
        """Block forever, ticking on the real clock every ``interval`` seconds."""
        self._grace_until = self._clock() + self.grace_seconds
        logger.info(
            "gossip loop started (interval=%.2fs, grace=%.1fs)", interval, self.grace_seconds
        )
        while True:
            self.tick(self._clock())
            time.sleep(interval)

    def on_ack(self, peer: NodeId) -> None:
        """A probe of ``peer`` was acked; drop its pending entry so it can be re-probed."""
        self._pending.pop(peer, None)
