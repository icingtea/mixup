"""Node entrypoint: run as the leader or an expert node, with failover.

Run inside every container. The runtime environment (container name, startup layout,
prompt, token budget) is read from the single MIXUP_NODE_ENV env var. Every node serves a
persistent FastAPI app (health, leader, routing, gossip, generate) and a raw-TCP expert
data plane; a background control thread boots the node and runs the SWIM gossip loop.
"""

import logging
import struct
import threading
import time
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from transformers import GraniteMoeConfig

from mixup_core.constants import EXPERT_PORT, INTERNAL_PORT, MODEL_DIR
from mixup_core.election import leader_projection
from mixup_core.env import load_node_env
from mixup_core.logging_config import configure_logging
from mixup_core.runtime import ClusterView, RoutingTable, apply_routing
from mixup_core.topology import cluster_view_from_config, validate_partition
from mixup_core.types import GenerateRequest, GenerateResponse

from .memory import TieredExpertStore
from .node import Node
from .serialization import ExpertRequest, ExpertResponse
from .transport import TCPServer

logger = logging.getLogger(__name__)


def _model_config() -> GraniteMoeConfig:
    return GraniteMoeConfig.from_pretrained(MODEL_DIR)


def build_app(node: Node) -> FastAPI:
    """A FastAPI app exposing health, leader, routing, and the expert endpoint."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "ready": node.ready}

    @app.get("/leader")
    async def leader() -> dict:
        return leader_projection(node.engine.view).model_dump()

    @app.post("/gossip")
    async def gossip(request: Request) -> dict:
        node.handle_gossip(await request.json())
        return {"ok": True}

    @app.post("/kv")
    async def kv(request: Request) -> dict:
        node.store_kv(await request.body())
        return {"ok": True}

    @app.post("/leader")
    async def leader_announce(request: Request) -> dict:
        data = await request.json()
        node.accept_leader(data["node_id"], data["epoch"])
        return {"ok": True}

    @app.post("/routing")
    async def routing(request: Request) -> dict:
        table = RoutingTable.model_validate(await request.json())
        current = node.engine.view.routing
        if current is not None and not table.is_newer_than(current):
            logger.info(
                "routing: ignoring stale table epoch=%d (have %d)",
                table.epoch,
                current.epoch,
            )
            return {"ok": True, "stale": True}
        node.engine.mutate(lambda v: apply_routing(v, table))
        if node.expert_group is not None:
            logger.info("routing: received table epoch=%d", table.epoch)
            node.expert_group.set_routing(table)
        return {"ok": True}

    @app.get("/routing")
    async def get_routing() -> dict:
        table = node.engine.view.routing
        if table is None:
            return {"epoch": None, "assignment": {}}
        return {
            "epoch": table.epoch,
            "assignment": {
                str(e): {"node": r.node_id, "slot": r.slot, "backup": r.backup}
                for e, r in table.assignment.items()
            },
        }

    @app.post("/join")
    async def join(request: Request) -> dict:
        if node.model is None:
            raise HTTPException(status_code=503, detail="not the leader")
        data = await request.json()
        node.rejoin(data["node_id"], data.get("incarnation"))
        return {"ok": True}

    @app.post("/replan")
    async def replan() -> dict:
        if node.model is None:
            raise HTTPException(status_code=503, detail="not the leader")
        table = node.replan()
        return {"ok": True, "epoch": table.epoch if table else None}

    @app.get("/stats")
    async def stats() -> dict:
        if node._trace is None:
            return {"total_tokens": 0}
        return {"total_tokens": node._trace.tokens}

    @app.get("/store")
    async def store() -> dict:
        group = node.expert_group
        if group is None:
            return {"type": "none"}
        if isinstance(group, TieredExpertStore):
            return {
                "type": "tiered",
                "owned": group.experts,
                "resident": group.resident_experts(),
                "execution_tier": group.execution_tier,
            }
        return {"type": "resident", "experts": group.experts}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict:
        if node.model is None:
            raise HTTPException(status_code=503, detail="not the leader")
        text, tokens, elapsed, dense, dispatch = node.generate(
            request.prompt, request.max_new_tokens
        )
        return GenerateResponse(
            text=text,
            tokens=tokens,
            elapsed_seconds=elapsed,
            dense_seconds=dense,
            dispatch_seconds=dispatch,
        ).model_dump()

    return app


def _expert_callback(node: Node) -> Callable[[bytes], bytes]:
    """Build the raw-TCP data-plane handler: ``[layer][body] -> [server_seconds][out]``."""

    def handle(payload: bytes) -> bytes:
        (layer,) = struct.unpack(">I", payload[:4])
        if node.expert_group is None:
            raise RuntimeError("not an expert node")
        server_start = time.perf_counter()
        req = ExpertRequest.from_bytes(payload[4:])
        out = node.expert_group.forward(
            layer, req.hidden_states, req.top_k_index, req.top_k_weights
        )
        body = ExpertResponse(out=out).to_bytes()
        return struct.pack(">d", time.perf_counter() - server_start) + body

    return handle


def start() -> None:
    """Deserialize MIXUP_NODE_ENV and start the server + control loop."""
    configure_logging()
    env = load_node_env()
    model_config = _model_config()
    validate_partition(env.config, model_config.num_local_experts or 0)
    view = cluster_view_from_config(env.config)
    # Fresh incarnation per boot (time-based): a restarted node is distinguishable from
    # its prior self, so peers revive it instead of treating its revival as stale gossip.
    me = view.members[env.container_name]
    view = ClusterView(
        members={
            **view.members,
            env.container_name: me.model_copy(update={"incarnation": int(time.time())}),
        },
        routing=view.routing,
    )

    node = Node(env, model_config, view)
    logger.info("%s boot: role=%s", env.container_name, node.role.value)

    # Expert data plane: a raw-TCP server, bound synchronously so it is listening before
    # the node reports ready. The control plane stays on FastAPI/HTTP.
    tcp_server = TCPServer("0.0.0.0", EXPERT_PORT, _expert_callback(node))
    threading.Thread(target=tcp_server.serve_forever, daemon=True).start()
    threading.Thread(target=node.run, daemon=True).start()
    uvicorn.run(build_app(node), host="0.0.0.0", port=INTERNAL_PORT, log_level="warning")


if __name__ == "__main__":
    start()
