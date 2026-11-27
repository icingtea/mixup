"""mixup-repl: prompt the running mixup cluster (CLI benchmark client).

Discovers the current leader and generates with failover-aware retry. If the leader dies
mid-request (connection drops), the client re-discovers the leader (highest epoch, across
every node address) and retries against the new leader.
"""

import argparse
import logging
import sys
import time

import httpx

from mixup_core.hosts import node_urls
from mixup_core.logging_config import configure_logging
from mixup_core.startup_config import load_startup_config
from mixup_core.types import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


def pick_leader(projections: list[dict]) -> str | None:
    """Given ``/leader`` projections, return the leader node id with the highest epoch."""
    best: str | None = None
    best_epoch = -1
    for proj in projections:
        leader = proj.get("leader")
        epoch = proj.get("epoch", 0)
        if leader is not None and epoch > best_epoch:
            best_epoch = epoch
            best = leader
    return best


class ClusterClient:
    """Discovers the leader and generates with failover-aware retry across node addresses."""

    def __init__(self, urls: dict[str, str], discover_timeout: float = 2.0) -> None:
        self.urls = urls
        self.discover_timeout = discover_timeout

    def discover_leader(self) -> str | None:
        """Probe every node's ``/leader``; return the node id with the highest epoch."""
        projections: list[dict] = []
        for url in self.urls.values():
            try:
                projections.append(httpx.get(f"{url}/leader", timeout=self.discover_timeout).json())
            except httpx.HTTPError:
                continue
        leader = pick_leader(projections)
        return leader if leader in self.urls else None

    def wait_ready(self, url: str, timeout: float = 30.0) -> bool:
        """Poll ``/health`` until the node reports ``ready=True``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=2).json().get("ready"):
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1)
        return False

    def generate(
        self,
        request: GenerateRequest,
        retries: int = 10,
        backoff: float = 2.0,
    ) -> GenerateResponse | None:
        """Generate, re-discovering the leader and retrying on connection drops.

        Returns the response, or ``None`` if no leader became reachable within ``retries``.
        """
        last_leader: str | None = None
        for attempt in range(retries):
            leader = self.discover_leader()
            if leader is None:
                logger.warning("no leader discovered (attempt %d/%d)", attempt + 1, retries)
                time.sleep(backoff)
                continue
            if leader != last_leader:
                logger.info("using leader %s", leader)
                last_leader = leader
            url = self.urls[leader]
            if not self.wait_ready(url):
                logger.warning("leader %s not ready (attempt %d/%d)", leader, attempt + 1, retries)
                continue
            try:
                # No timeout: generation is long; a killed leader raises an HTTPError on
                # the dropped connection, which we catch and retry against the new leader.
                response = httpx.post(f"{url}/generate", json=request.model_dump(), timeout=None)
                response.raise_for_status()
                return GenerateResponse(**response.json())
            except httpx.HTTPError as exc:
                logger.warning(
                    "leader %s failed (%s); waiting for failover...", leader, exc
                )
                time.sleep(backoff)
        return None


def _report(
    text: str,
    tokens: int,
    elapsed: float,
    dense: float | None = None,
    dispatch: float | None = None,
) -> None:
    print(text)
    if elapsed <= 0:
        return
    tok_s = tokens / elapsed
    line = f"[{tokens} tokens in {elapsed:.3f}s ({tok_s:.2f} tok/s"
    if dense is not None and dispatch is not None:
        line += f" | dense {dense:.3f}s dispatch {dispatch:.3f}s"
    line += "]"
    print(line, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Prompt the mixup cluster.")
    parser.add_argument("prompt", nargs="?", help="the prompt to send")
    parser.add_argument(
        "--config", default="startup_config.json", help="startup_config.json or lab_config.json"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="generation length (defaults to the config's max_new_tokens, else 500)",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="per-node discovery timeout (seconds)"
    )
    parser.add_argument(
        "--retries", type=int, default=30, help="generation attempts across failovers"
    )
    parser.add_argument(
        "--backoff", type=float, default=2.0, help="seconds to wait between attempts"
    )
    args = parser.parse_args(argv)

    if args.prompt is None:
        parser.error("a prompt is required")

    config = load_startup_config(args.config)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else getattr(config, "max_new_tokens", 500)
    )
    client = ClusterClient(node_urls(config), discover_timeout=args.timeout)
    request = GenerateRequest(prompt=args.prompt, max_new_tokens=max_new_tokens)

    result = client.generate(request, retries=args.retries, backoff=args.backoff)
    if result is None:
        print("could not reach a leader after retries", file=sys.stderr)
        return 1

    _report(
        result.text,
        result.tokens,
        result.elapsed_seconds,
        result.dense_seconds,
        result.dispatch_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
