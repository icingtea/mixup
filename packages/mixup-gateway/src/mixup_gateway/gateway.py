"""mixup-gateway: a stable HTTP front door for the cluster.

The gateway is independent of any single node. Clients connect to it once and it keeps
their requests alive across a leader failover: it re-discovers the leader (highest
epoch) and re-issues the request, so a leader swap is invisible to the client.

It runs host-side (no torch, no GPU): it only needs the node URLs, which it derives
from the same ``startup_config.json`` the orchestrator uses.
"""

import argparse
import logging
import time

import httpx
from fastapi import FastAPI, HTTPException

from mixup_core.hosts import node_urls
from mixup_core.logging_config import configure_logging
from mixup_core.startup_config import load_startup_config
from mixup_core.types import GenerateRequest, GenerateResponse, StartupConfig

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8999


class Gateway:
    """Routes generation requests to the current leader, retrying across failover."""

    def __init__(self, config: StartupConfig, discover_timeout: float = 2.0) -> None:
        self.urls = node_urls(config)
        self.discover_timeout = discover_timeout

    def discover_leader(self) -> str | None:
        """Return the node id of the current leader (highest epoch), or ``None``."""
        best: str | None = None
        best_epoch = -1
        for url in self.urls.values():
            try:
                projection = httpx.get(f"{url}/leader", timeout=self.discover_timeout).json()
            except httpx.HTTPError:
                continue
            leader = projection.get("leader")
            epoch = projection.get("epoch", 0)
            if leader is not None and epoch > best_epoch:
                best_epoch = epoch
                best = leader
        return best

    def _wait_ready(self, url: str, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=2).json().get("ready"):
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1)
        return False

    def generate(self, request: GenerateRequest, retries: int = 5) -> GenerateResponse | None:
        """Forward to the current leader; re-discover and retry on failover.

        Returns ``None`` if no leader becomes ready within ``retries`` attempts.
        """
        for attempt in range(retries):
            leader = self.discover_leader()
            if leader is None or leader not in self.urls:
                logger.warning(
                    "gateway: no leader discovered (attempt %d/%d)", attempt + 1, retries
                )
                time.sleep(2)
                continue
            url = self.urls[leader]
            if not self._wait_ready(url):
                logger.warning(
                    "gateway: leader %s not ready (attempt %d/%d)", leader, attempt + 1, retries
                )
                continue
            try:
                response = httpx.post(f"{url}/generate", json=request.model_dump(), timeout=None)
                response.raise_for_status()
                return GenerateResponse(**response.json())
            except httpx.HTTPError as exc:
                logger.warning(
                    "gateway: leader %s failed (%s); waiting for failover...", leader, exc
                )
                time.sleep(2)
        return None


def build_app(gateway: Gateway) -> FastAPI:
    """The gateway's HTTP surface (same shape as the leader's, minus the expert routes)."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/leader")
    async def leader() -> dict:
        return {"leader": gateway.discover_leader()}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict:
        result = gateway.generate(request)
        if result is None:
            raise HTTPException(status_code=503, detail="no leader available after retries")
        return result.model_dump()

    return app


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the mixup gateway.")
    parser.add_argument("config", nargs="?", default="startup_config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="per-node discovery timeout (seconds)"
    )
    args = parser.parse_args(argv)

    config = load_startup_config(args.config)
    gateway = Gateway(config, discover_timeout=args.timeout)
    logger.info("gateway listening on %s:%d", args.host, args.port)

    # Imported lazily so the gateway's core is testable without uvicorn.
    import uvicorn

    uvicorn.run(build_app(gateway), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
