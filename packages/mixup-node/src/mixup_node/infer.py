"""Node entrypoint: run as the leader or an expert node, with failover.

Run inside every container. The runtime environment (container name, startup layout,
prompt, token budget) is read from the single MIXUP_NODE_ENV env var. Every node serves a
persistent FastAPI app (health, leader, routing, expert); a background control thread runs
generation (leader) or the leader watchdog (follower).
"""

import logging
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from transformers import GraniteMoeConfig

from mixup_core.constants import INTERNAL_PORT, MODEL_DIR
from mixup_core.election import leader_projection
from mixup_core.env import load_node_env
from mixup_core.logging_config import configure_logging
from mixup_core.runtime import RoutingTable
from mixup_core.topology import cluster_view_from_config, validate_partition
from mixup_core.types import GenerateRequest, GenerateResponse

from .node import Node
from .serialization import ExpertRequest, ExpertResponse

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
        return leader_projection(node.view).model_dump()

    @app.post("/leader")
    async def leader_announce(request: Request) -> dict:
        data = await request.json()
        node.accept_leader(data["node_id"], data["epoch"])
        return {"ok": True}

    @app.post("/routing")
    async def routing(request: Request) -> dict:
        table = RoutingTable.model_validate(await request.json())
        if node.expert_group is not None:
            logger.info("routing: received table epoch=%d", table.epoch)
            node.expert_group.set_routing(table)
        return {"ok": True}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict:
        if node.model is None:
            raise HTTPException(status_code=503, detail="not the leader")
        text, tokens, elapsed = node.generate(request.prompt, request.max_new_tokens)
        return GenerateResponse(
            text=text, tokens=tokens, elapsed_seconds=elapsed
        ).model_dump()

    @app.post("/{layer}")
    async def run(layer: int, request: Request) -> Response:
        if node.expert_group is None:
            return Response(status_code=404)
        req = ExpertRequest.from_bytes(await request.body())
        out = node.expert_group.forward(
            layer, req.hidden_states, req.top_k_index, req.top_k_weights
        )
        return Response(ExpertResponse(out=out).to_bytes())

    return app


def start() -> None:
    """Deserialize MIXUP_NODE_ENV and start the server + control loop."""
    configure_logging()
    env = load_node_env()
    model_config = _model_config()
    validate_partition(env.config, model_config.num_local_experts or 0)
    view = cluster_view_from_config(env.config)

    node = Node(env, model_config, view)
    logger.info("%s boot: role=%s", env.container_name, node.role.value)

    threading.Thread(target=node.run, daemon=True).start()
    uvicorn.run(build_app(node), host="0.0.0.0", port=INTERNAL_PORT, log_level="warning")


if __name__ == "__main__":
    start()
