"""HTTP transport for the gossip protocol: a thin FastAPI layer over GossipEngine.

Deliberately torch-free: this module imports only ``mixup_core`` and FastAPI, so the
gossip endpoints can be tested without loading torch. The expert-serving path (server.py)
is a separate module that does import torch.
"""

import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mixup_core.election import leader_projection
from mixup_core.gossip import GossipMsg, decode_gossip, encode_gossip
from mixup_core.gossip_engine import GossipEngine
from mixup_core.runtime import ClusterView, RoutingTable, apply_routing
from mixup_core.types import NodeId

Clock = Callable[[], float]


class GossipServer:
    """A node's live view + gossip engine, made thread-safe for the HTTP handlers."""

    def __init__(
        self,
        node_id: NodeId,
        view: ClusterView,
        *,
        ping_timeout: float,
        suspect_timeout: float,
        indirect_k: int,
        clock: Clock = time.monotonic,
    ) -> None:
        self.engine = GossipEngine(
            node_id,
            view,
            ping_timeout=ping_timeout,
            suspect_timeout=suspect_timeout,
            indirect_k=indirect_k,
        )
        self._clock = clock
        self._lock = threading.Lock()

    def handle(self, msg: GossipMsg) -> list[tuple[NodeId | None, dict]]:
        """Handle an inbound message; return the (destination, message) replies to send."""
        with self._lock:
            replies = self.engine.handle(msg, now=self._clock())
        return [(dest, encode_gossip(reply)) for dest, reply in replies]

    def apply_routing(self, table: RoutingTable) -> None:
        with self._lock:
            self.engine.view = apply_routing(self.engine.view, table)

    def routing(self) -> dict | None:
        with self._lock:
            routing = self.engine.view.routing
            return routing.model_dump() if routing else None

    def projection(self) -> dict:
        with self._lock:
            return leader_projection(self.engine.view).model_dump()


def create_app(server: GossipServer) -> FastAPI:
    """A FastAPI app exposing the gossip, routing, leader, and health endpoints."""
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/gossip")
    async def gossip(request: Request) -> JSONResponse:
        msg = decode_gossip(await request.json())
        replies = server.handle(msg)
        return JSONResponse({"replies": [{"to": dest, "msg": reply} for dest, reply in replies]})

    @app.post("/routing")
    async def routing(request: Request) -> dict:
        table = RoutingTable.model_validate(await request.json())
        server.apply_routing(table)
        return {"ok": True}

    @app.get("/routing")
    def get_routing() -> dict:
        return server.routing() or {}

    @app.get("/leader")
    def leader() -> dict:
        return server.projection()

    return app
