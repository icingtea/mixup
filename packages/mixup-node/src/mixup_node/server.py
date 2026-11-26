"""Expert-side HTTP server: answer expert requests and apply routing-table updates."""

import uvicorn
from fastapi import FastAPI, Request, Response

from mixup_core.runtime import RoutingTable

from .model import ExpertGroup
from .serialization import ExpertRequest, ExpertResponse


def create_app(group: ExpertGroup) -> FastAPI:
    """A FastAPI app that runs `group`'s experts and accepts routing-table swaps."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/routing")
    async def routing(request: Request) -> dict:
        table = RoutingTable.model_validate(await request.json())
        group.set_routing(table)
        return {"ok": True}

    @app.post("/{layer}")
    async def run(layer: int, request: Request) -> Response:
        req = ExpertRequest.from_bytes(await request.body())
        out = group.forward(layer, req.hidden_states, req.top_k_index, req.top_k_weights)
        return Response(ExpertResponse(out=out).to_bytes())

    return app


def serve(group: ExpertGroup, port: int) -> None:
    """Serve `group` on 0.0.0.0:port until interrupted."""
    uvicorn.run(create_app(group), host="0.0.0.0", port=port, log_level="warning")
