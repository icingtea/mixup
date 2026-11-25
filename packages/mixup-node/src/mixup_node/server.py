"""Expert-side HTTP server: answer expert requests from the central node."""

import uvicorn
from fastapi import FastAPI, Request, Response

from .model import ExpertGroup
from .serialization import ExpertRequest, ExpertResponse


def create_app(group: ExpertGroup) -> FastAPI:
    """A FastAPI app that runs `group`'s experts for a given layer."""
    app = FastAPI()

    @app.post("/{layer}")
    async def run(layer: int, request: Request) -> Response:
        req = ExpertRequest.from_bytes(await request.body())
        out = group.forward(layer, req.hidden_states, req.top_k_index, req.top_k_weights)
        return Response(ExpertResponse(out=out).to_bytes())

    return app


def serve(group: ExpertGroup, port: int) -> None:
    """Serve `group` on 0.0.0.0:port until interrupted."""
    uvicorn.run(create_app(group), host="0.0.0.0", port=port, log_level="warning")
