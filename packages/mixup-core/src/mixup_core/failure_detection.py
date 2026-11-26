"""SWIM failure-detection timing logic (pure; time is injected explicitly).

The transport (mixup-node) owns the network: it pings peers on a schedule and calls
these methods with explicit timestamps. This class only tracks the per-peer protocol
state and decides the next action — direct probe → indirect probe → suspect → dead —
which is what the transport then performs. Keeping it free of clock and socket calls
makes the whole detection protocol unit-testable with synthetic time.
"""

from enum import StrEnum

from .types import NodeId


class Action(StrEnum):
    """What the transport should do after a probe wait expires."""

    PINGREQ = "pingreq"
    """Direct probe failed; send indirect probes via k random peers."""

    SUSPECT = "suspect"
    """Indirect probes failed; gossip Suspect for this peer."""

    DEAD = "dead"
    """Suspicion expired without revival; gossip Dead for this peer."""

    WAIT = "wait"
    """No action this tick."""


_AWAITING_ACK = "awaiting_ack"
_AWAITING_INDIRECT = "awaiting_indirect"
_SUSPECT = "suspect"
_DEAD = "dead"


class FailureDetector:
    """Per-peer SWIM probe state, advanced by explicit ``expire(peer, now)`` calls.

    The transitions mirror SWIM: a failed direct ping triggers indirect probing, a
    failed indirect probe marks the peer suspect, and an un-revived suspect is
    declared dead after ``suspect_timeout``.
    """

    def __init__(self, ping_timeout: float, suspect_timeout: float) -> None:
        self.ping_timeout = ping_timeout
        self.suspect_timeout = suspect_timeout
        self._state: dict[NodeId, str] = {}
        self._since: dict[NodeId, float] = {}

    def ping(self, peer: NodeId, now: float) -> None:
        """Record that a direct ping was just sent to ``peer``."""
        self._state[peer] = _AWAITING_ACK
        self._since[peer] = now

    def ack(self, peer: NodeId) -> bool:
        """Peer responded (directly or indirectly); clear any probe/suspicion.

        Returns True if the peer was previously suspect or dead (i.e. it just revived),
        which tells the transport to gossip an ``Alive``.
        """
        was_down = self._state.get(peer) in (_SUSPECT, _DEAD)
        self._state.pop(peer, None)
        self._since.pop(peer, None)
        return was_down

    def expire(self, peer: NodeId, now: float) -> Action:
        """A probe wait elapsed for ``peer``. Return the next action for the transport.

        Ignores the ``ping_timeout`` in the timing math because the transport only calls
        this once a wait has actually elapsed; ``ping_timeout`` is retained for callers
        that want to schedule from it.
        """
        state = self._state.get(peer)
        if state is None or state is _DEAD:
            return Action.WAIT
        if state is _AWAITING_ACK:
            self._state[peer] = _AWAITING_INDIRECT
            self._since[peer] = now
            return Action.PINGREQ
        if state is _AWAITING_INDIRECT:
            self._state[peer] = _SUSPECT
            self._since[peer] = now
            return Action.SUSPECT
        # state is _SUSPECT
        if now - self._since[peer] >= self.suspect_timeout:
            self._state[peer] = _DEAD
            return Action.DEAD
        return Action.WAIT
