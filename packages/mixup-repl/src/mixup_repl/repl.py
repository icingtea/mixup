"""mixup-repl: prompt the running cluster from the host.

Discovers the current leader by probing every node's ``GET /leader`` (picking the
highest epoch, so it stays correct across a failover), then sends the prompt to the
leader's ``POST /generate``.
"""

import argparse
import sys

import httpx

from mixup_core.constants import EXTERNAL_PORT_BASE
from mixup_core.startup_config import load_startup_config
from mixup_core.types import GenerateRequest, GenerateResponse, StartupConfig


def node_urls(config: StartupConfig) -> dict[str, str]:
    """Map each node to its host URL, mirroring the orchestrator's port assignment."""
    urls = {config.initial_leader: f"http://localhost:{EXTERNAL_PORT_BASE}"}
    for offset, name in enumerate(config.experts):
        urls[name] = f"http://localhost:{EXTERNAL_PORT_BASE + offset + 1}"
    return urls


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


def discover_leader(urls: dict[str, str], timeout: float) -> str | None:
    """Probe every node's ``/leader``; return the leader's node id (highest epoch)."""
    projections: list[dict] = []
    for url in urls.values():
        try:
            projections.append(httpx.get(f"{url}/leader", timeout=timeout).json())
        except httpx.HTTPError:
            continue
    return pick_leader(projections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt the running mixup cluster.")
    parser.add_argument("prompt", help="the prompt to send")
    parser.add_argument(
        "--config", default="startup_config.json", help="path to startup_config.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="per-node discovery timeout (seconds)"
    )
    args = parser.parse_args(argv)

    config = load_startup_config(args.config)
    urls = node_urls(config)

    leader_id = discover_leader(urls, args.timeout)
    if leader_id is None:
        print("no leader found (is the cluster running?)", file=sys.stderr)
        return 1
    leader_url = urls[leader_id]

    request = GenerateRequest(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
    try:
        response = httpx.post(
            f"{leader_url}/generate", json=request.model_dump(), timeout=None
        )
        response.raise_for_status()
        result = GenerateResponse(**response.json())
    except httpx.HTTPError as exc:
        print(f"error talking to {leader_url}: {exc}", file=sys.stderr)
        return 1

    print(result.text)
    if result.elapsed_seconds > 0:
        tok_s = result.tokens / result.elapsed_seconds
        print(
            f"[{result.tokens} tokens in {result.elapsed_seconds:.3f}s "
            f"({tok_s:.2f} tok/s)]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
