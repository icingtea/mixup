"""The gossip protocol engine: a node's side of SWIM, tying message handling to failure
detection and the indirect-probe relay.

Pure and time-injected: every method takes an explicit ``now`` and returns the outbound
messages the transport should send (or ``None`` for broadcast). The transport owns the
sockets, the ping schedule, and the HTTP endpoints; this class owns the decisions.

References: [SWIM], [Serf-gossip].
"""

import threading
from collections.abc import Callable

from .failure_detection import Action, FailureDetector
from .gossip import Ack, Alive, Dead, GossipMsg, Ping, PingReq, Suspect, apply_gossip
from .runtime import ClusterView, MemberState
from .types import MemberStatus, NodeId


class GossipEngine:
    """One node's SWIM protocol state and message-handling decisions.

    Thread-safe: the view is the single source of truth shared by the gossip loop, the
    HTTP gossip handler, and the node's control thread. All mutations go through one lock;
    ``mutate`` is how the node applies its own view changes (promote, rebalance, ...).
    """

    def __init__(
        self,
        node_id: NodeId,
        view: ClusterView,
        *,
        ping_timeout: float,
        suspect_timeout: float,
        indirect_k: int,
    ) -> None:
        self.node_id = node_id
        self.view = view
        self.detector = FailureDetector(ping_timeout, suspect_timeout)
        self.indirect_k = indirect_k
        # target -> requesters that asked us to indirectly probe it
        self._relays: dict[NodeId, list[NodeId]] = {}
        self._lock = threading.Lock()

    def mutate(self, fn: Callable[[ClusterView], ClusterView]) -> ClusterView:
        """Apply ``fn`` to the view under the lock and return the new view."""
        with self._lock:
            self.view = fn(self.view)
            return self.view

    # -- helpers -----------------------------------------------------------

    def _self(self) -> MemberState:
        return self.view.members[self.node_id]

    def _incarnation(self, peer: NodeId) -> int:
        member = self.view.members.get(peer)
        return member.incarnation if member else 0

    def _self_ping(self) -> Ping:
        me = self._self()
        return Ping(
            node_id=self.node_id,
            incarnation=me.incarnation,
            epoch=me.epoch,
            kv_seq=me.kv_seq,
        )

    def _merge_peer(self, node_id: NodeId, incarnation: int, epoch: int, kv_seq: int) -> None:
        """Adopt a peer's fresher (incarnation, epoch, kv_seq) seen in a gossip message.

        This is how a node's KV-cache freshness (``kv_seq``) and leadership epoch reach the
        rest of the cluster: Ping/Ack carry them, and the election ranks by ``kv_seq``
        first, so without this the winner can't prefer the most up-to-date node. A *higher*
        incarnation is a restart, so we also revive the peer to ALIVE (its messages prove
        it is back) — this is what makes expert rejoin work.
        """
        member = self.view.members.get(node_id)
        if member is None or incarnation < member.incarnation:
            return  # unknown, or a stale (older-incarnation) message
        if incarnation > member.incarnation:
            # A restart: adopt the new incarnation + epoch/kv_seq, and revive it.
            updated = member.model_copy(
                update={
                    "incarnation": incarnation,
                    "epoch": epoch,
                    "kv_seq": kv_seq,
                    "status": MemberStatus.ALIVE,
                }
            )
        elif epoch > member.epoch or kv_seq > member.kv_seq:
            updated = member.model_copy(
                update={"epoch": max(member.epoch, epoch), "kv_seq": max(member.kv_seq, kv_seq)}
            )
        else:
            return
        self.view = ClusterView(
            members={**self.view.members, node_id: updated}, routing=self.view.routing
        )

    # -- outbound triggers -------------------------------------------------

    def probe(self, peer: NodeId, now: float) -> Ping:
        """Begin a direct probe of ``peer``; return the Ping to send."""
        with self._lock:
            self.detector.ping(peer, now)
            return self._self_ping()

    def on_timeout(self, peer: NodeId, now: float) -> tuple[Action, GossipMsg | None]:
        """A probe wait expired for ``peer``. Return the action and any broadcast message.

        ``SUSPECT``/``DEAD`` carry a message to broadcast (and update our own view);
        ``PINGREQ`` and ``WAIT`` carry ``None`` (the transport picks the indirect peers).
        """
        with self._lock:
            action = self.detector.expire(peer, now)
            if action is Action.SUSPECT:
                msg: GossipMsg = Suspect(
                    node_id=peer, incarnation=self._incarnation(peer), by_node=self.node_id
                )
                self.view = apply_gossip(self.view, msg)
                return action, msg
            if action is Action.DEAD:
                msg = Dead(
                    node_id=peer, incarnation=self._incarnation(peer), by_node=self.node_id
                )
                self.view = apply_gossip(self.view, msg)
                return action, msg
            return action, None

    # -- inbound handling --------------------------------------------------

    def handle(self, msg: GossipMsg, now: float) -> list[tuple[NodeId | None, GossipMsg]]:
        """Process an inbound message; return ``(destination, message)`` pairs to send.

        A ``destination`` of ``None`` means broadcast to all peers.
        """
        with self._lock:
            if isinstance(msg, Ping):
                self._merge_peer(msg.node_id, msg.incarnation, msg.epoch, msg.kv_seq)
                me = self._self()
                ack = Ack(
                    node_id=self.node_id,
                    incarnation=me.incarnation,
                    epoch=me.epoch,
                    kv_seq=me.kv_seq,
                )
                return [(msg.node_id, ack)]

            if isinstance(msg, Ack):
                self._merge_peer(msg.node_id, msg.incarnation, msg.epoch, msg.kv_seq)
                out: list[tuple[NodeId | None, GossipMsg]] = []
                if self.detector.ack(msg.node_id):
                    alive = Alive(
                        node_id=msg.node_id,
                        incarnation=msg.incarnation,
                        epoch=msg.epoch,
                    )
                    # Revive in our own view too (not just broadcast to peers).
                    self.view = apply_gossip(self.view, alive)
                    out.append((None, alive))
                for requester in self._relays.pop(msg.node_id, []):
                    out.append(
                        (
                            requester,
                            Ack(
                                node_id=msg.node_id,
                                incarnation=msg.incarnation,
                                epoch=msg.epoch,
                                kv_seq=msg.kv_seq,
                                routing_epoch=msg.routing_epoch,
                            ),
                        )
                    )
                return out

            if isinstance(msg, PingReq):
                self._relays.setdefault(msg.target, []).append(msg.from_node)
                # Just forward a probe to the target on the requester's behalf. Do NOT
                # touch self.detector for the target: that would reset our own SUSPECT
                # state (we may already be failing to reach it ourselves), and every node
                # pinging each other with PingReq would keep the SUSPECT->DEAD transition
                # from ever firing.
                return [(msg.target, self._self_ping())]

            if isinstance(msg, Alive):
                self._merge_peer(msg.node_id, msg.incarnation, msg.epoch, 0)
                self.view = apply_gossip(self.view, msg)
                return []

            if isinstance(msg, (Suspect, Dead)):
                self.view = apply_gossip(self.view, msg)
                return []

            return []
