"""Node entrypoint: run as the leader or an expert node, with failover.

Run inside every container. The runtime environment (container name, startup layout,
prompt, token budget) is read from the single MIXUP_NODE_ENV env var. Every node serves a
persistent FastAPI app (health, leader, routing, gossip, generate, and the ``/expert``
data-plane endpoint); a background control thread boots the node and runs the SWIM gossip
loop.
"""

import logging
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
from transformers import GraniteMoeConfig

from mixup_core.constants import INTERNAL_PORT, MODEL_DIR
from mixup_core.election import leader_projection
from mixup_core.env import load_node_env
from mixup_core.logging_config import configure_logging
from mixup_core.runtime import ClusterView, RoutingTable
from mixup_core.topology import cluster_view_from_config, validate_partition
from mixup_core.types import GenerateRequest, GenerateResponse

from .memory import TieredExpertStore
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
        return leader_projection(node.engine.view).model_dump()

    @app.post("/gossip")
    async def gossip(request: Request) -> dict:
        node.handle_gossip(await request.json())
        return {"ok": True}

    @app.post("/kv")
    async def kv(request: Request) -> dict:
        node.store_kv(await request.body())
        return {"ok": True}

    @app.post("/commit")
    async def commit(request: Request) -> dict:
        data = await request.json()
        acked = node.store_commit(data)
        return {"ok": True, "acked": acked}

    @app.post("/predictor")
    async def predictor(request: Request) -> dict:
        node.store_predictor(await request.json())
        return {"ok": True}

    @app.get("/predictor")
    async def predictor_info() -> dict:
        return node.predictor_info()

    @app.get("/experts")
    async def expert_stats() -> dict:
        return node.expert_stats()

    @app.post("/leader")
    async def leader_announce(request: Request) -> dict:
        data = await request.json()
        node.accept_leader(data["node_id"], data["epoch"])
        return {"ok": True}

    @app.post("/routing")
    async def routing(request: Request) -> dict:
        table = RoutingTable.model_validate(await request.json())
        # Adopt atomically: the view is updated under the engine lock, so a stale table
        # (older epoch) can never overwrite a newer one.
        if not node.engine.adopt_routing(table):
            logger.info("routing: ignoring stale table epoch=%d", table.epoch)
            return {"ok": True, "stale": True}
        if node.expert_group is not None:
            logger.info("routing: received table epoch=%d", table.epoch)
            # The expert reload is GIL-heavy and would block the event loop (and thus the
            # /gossip ping acks) if run here; offload it so failure detection stays live.
            await run_in_threadpool(node.expert_group.set_routing, table)
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
        # rejoin rebalances + reloads experts (GIL-heavy); offload so /gossip stays live.
        await run_in_threadpool(node.rejoin, data["node_id"], data.get("incarnation"))
        return {"ok": True}

    @app.post("/replan")
    async def replan() -> dict:
        if node.model is None:
            raise HTTPException(status_code=503, detail="not the leader")
        table = await run_in_threadpool(node.replan)
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

    @app.get("/metrics")
    async def metrics() -> dict:
        """Live resource perception: smoothed CPU/memory/GPU, profile, routing, stats."""
        profile = node.env.profiles.get(node.name)
        table = node.engine.view.routing
        residency = None
        if isinstance(node.expert_group, TieredExpertStore):
            residency = node.expert_group.residency_info()
        return {
            "node": node.name,
            "role": node.role.value,
            "execution": profile.execution.value if profile else None,
            "resources": node.monitor.snapshot(),
            "profile": profile.model_dump() if profile else None,
            "residency": residency,
            "stats": {"total_tokens": node._trace.tokens if node._trace else 0},
            "routing_epoch": table.epoch if table else None,
            "owned_experts": node.expert_group.experts if node.expert_group else [],
        }

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

    @app.post("/expert")
    async def expert(request: Request) -> Response:
        """Expert data plane: run one layer's experts on this node's share.

        Body is the ``ExpertRequest`` safetensors bytes; the layer id rides in the
        ``X-Layer`` header (the binary body stays a pure safetensors blob). The GIL-heavy
        deserialize + forward + serialize runs off the event loop so /gossip stays live.
        """
        if node.expert_group is None:
            raise HTTPException(status_code=503, detail="not an expert node")
        layer = int(request.headers.get("X-Layer", "0"))
        body = await request.body()
        return await run_in_threadpool(_expert_handle, node, layer, body)

    return app


def _expert_handle(node: Node, layer: int, body: bytes) -> Response:
    """Run the expert forward for ``layer`` and return the serialized ``ExpertResponse``."""
    server_start = time.perf_counter()
    req = ExpertRequest.from_bytes(body)
    # Prefetch the predicted next-layer experts (Phase 3 + 6) while we compute this
    # layer: the hint carries each candidate's activation probability, and the store
    # applies the paper's ``θ_pre`` threshold (driven by resource stability + free
    # memory) before deciding what to load.
    if req.prefetch.numel() > 0 and isinstance(node.expert_group, TieredExpertStore):
        nxt = layer + 1
        if nxt < node.model_config.num_hidden_layers:
            stability = node.monitor.snapshot().get("cpu_util_stability", 1.0)
            probs = (
                req.prefetch_probs.tolist()
                if req.prefetch_probs.numel() == req.prefetch.numel()
                else None
            )
            node.expert_group.prefetch(nxt, req.prefetch.tolist(), probs, stability)
    out = node.expert_group.forward(
        layer, req.hidden_states, req.top_k_index, req.top_k_weights
    )
    payload = ExpertResponse(out=out).to_bytes()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"X-Server-Time": str(time.perf_counter() - server_start)},
    )


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

    # The expert data plane is the /expert endpoint on the same FastAPI app (control +
    # data share one HTTP server); only the background control thread is separate.
    threading.Thread(target=node.run, daemon=True).start()
    uvicorn.run(build_app(node), host="0.0.0.0", port=INTERNAL_PORT, log_level="warning")


if __name__ == "__main__":
    start()
