"""Persistent single-process (no-distribution) chat backend, run inside a GPU node.

Reads JSON lines ``{"prompt": str, "max_new_tokens": int}`` from stdin, generates one
response per line (full model, single process), and writes one JSON line per response.
The model loads once, so consecutive messages are fast. Executed via
``docker exec -i <node> python -c <this file>`` with stdin/stdout kept open.
"""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "models/granite-3.1-1b-a400m-instruct"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
    except ValueError:
        continue
    prompt = data.get("prompt", "")
    max_new = int(data.get("max_new_tokens", 50))

    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    prompt_len = int(ids.shape[1])

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    elapsed = time.perf_counter() - start

    tokens = int(out.shape[1] - prompt_len)
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    result = {"ok": True, "text": text, "tokens": tokens, "elapsed_seconds": round(elapsed, 4)}
    print(json.dumps(result))
    sys.stdout.flush()
