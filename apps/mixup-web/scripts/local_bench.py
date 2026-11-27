"""Single-process (no-distribution) baseline, run inside a GPU node's container.

Reads ``{"prompt": str, "max_new_tokens": int}`` from ``sys.argv[1]``, loads the whole
model on the GPU, applies the Granite chat template, generates greedily, and prints one
line of JSON with the result. Executed via ``docker exec <node> python -c <this file>``.
"""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

data = json.loads(sys.argv[1])
prompt = data["prompt"]
max_new = int(data.get("max_new_tokens", 50))
model_dir = "models/granite-3.1-1b-a400m-instruct"

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

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
print(json.dumps({"ok": True, "text": text, "tokens": tokens, "elapsed_seconds": round(elapsed, 4)}))
sys.stdout.flush()
