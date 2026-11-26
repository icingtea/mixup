"""mixup-repl: prompt the mixup model, locally or on a running cluster.

Two modes:

* ``--local``: load the entire model into memory once (on the GPU when available)
  and answer prompts directly. No experts are distributed; this is a plain,
  single-process baseline.
* default: discover the current leader by probing every node's ``GET /leader``
  (picking the highest epoch, so it stays correct across a failover), then send
  the prompt to the leader's ``POST /generate``.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

from mixup_core.constants import EXTERNAL_PORT_BASE, MODEL_DIR
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


class LocalModel:
    """The whole model in memory, on the GPU, answering prompts in-process."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )
        print(f"[local] loaded {model_dir} on {self.device}", file=sys.stderr)

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
        """Greedy generation; returns (new text, tokens generated, seconds)."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_len = ids.shape[1]
        start = time.monotonic()
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.monotonic() - start
        tokens = out.shape[1] - prompt_len
        return self.tokenizer.decode(out[0][prompt_len:]), tokens, elapsed


def _report(text: str, tokens: int, elapsed: float) -> None:
    print(text)
    if elapsed > 0:
        print(
            f"[{tokens} tokens in {elapsed:.3f}s ({tokens / elapsed:.2f} tok/s)]",
            file=sys.stderr,
        )


def run_local(prompt: str | None, max_new_tokens: int, model_dir: Path) -> int:
    """Load the full model once, then answer one prompt (or an interactive loop)."""
    model = LocalModel(model_dir)
    if prompt is not None:
        _report(*model.generate(prompt, max_new_tokens))
        return 0

    for line in sys.stdin:
        line = line.strip()
        if line:
            _report(*model.generate(line, max_new_tokens))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt the mixup model.")
    parser.add_argument(
        "prompt", nargs="?", help="the prompt to send (omit with --local for a REPL loop)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="load the whole model locally (GPU if available); do not use the cluster",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR, help="path to the model directory"
    )
    parser.add_argument(
        "--config", default="startup_config.json", help="path to startup_config.json"
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="per-node discovery timeout (seconds)"
    )
    args = parser.parse_args(argv)

    if args.local:
        return run_local(args.prompt, args.max_new_tokens, args.model_dir)
    if args.prompt is None:
        parser.error("a prompt is required unless --local is given")

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

    _report(result.text, result.tokens, result.elapsed_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
