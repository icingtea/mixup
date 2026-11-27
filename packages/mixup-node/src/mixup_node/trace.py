"""``mixup-trace``: capture per-layer expert-selection traces from real generation.

Runs the model locally (like ``mixup-repl --local``) and records, for each *generated*
token, the global expert ids the router selected at every layer. Output is JSONL, one line
per generated token: ``{"token": <id>, "layers": [[e, ...], ...]}``.

The leader runs the same routers, so this trace is exactly what the placement engine's
``ActivationStats`` consumes (and what the simulator replays).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixup_core.constants import MODEL_DIR


def _record_layer_hooks(model, num_layers: int):
    """Attach a forward hook per layer's router, capturing its top_k_index output."""
    capture: dict[int, list[int]] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _args, output):
            # output = (router_logits, top_k_weights, top_k_index); index shape (tokens, top_k).
            capture[layer] = output[2][-1].tolist()

        return hook

    for layer, dec_layer in enumerate(model.model.layers[:num_layers]):
        handles.append(dec_layer.block_sparse_moe.router.register_forward_hook(make_hook(layer)))
    return capture, handles


def capture_trace(
    prompt: str,
    max_new_tokens: int,
    model_dir: Path = MODEL_DIR,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    num_layers = model.config.num_hidden_layers
    capture, handles = _record_layer_hooks(model, num_layers)
    try:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        trace: list[dict] = []
        with torch.no_grad():
            out = model(input_ids=prompt_ids, cache=None, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1)
            for _ in range(max_new_tokens):
                if nxt.item() == tokenizer.eos_token_id:
                    break
                out = model(input_ids=nxt.unsqueeze(0), cache=out.cache, use_cache=True)
                trace.append(
                    {"token": nxt.item(), "layers": [capture[layer] for layer in range(num_layers)]}
                )
                nxt = out.logits[:, -1, :].argmax(dim=-1)
        return trace
    finally:
        for handle in handles:
            handle.remove()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture expert-selection traces.")
    parser.add_argument("--prompt", default="What is mixture of experts?")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out", type=Path, default=None, help="write JSONL here (default stdout)")
    args = parser.parse_args(argv)

    trace = capture_trace(args.prompt, args.max_new_tokens, args.model_dir)
    lines = [json.dumps(t) for t in trace]
    if args.out:
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    else:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
