"""CPU expert-path smoke test (run inside a CPU-only container, no GPU).

Verifies that an ``ExpertGroup`` loads and runs its experts in fp32 on CPU, and that its
output matches a direct fp32 reference computation of the same experts (allowing for the
bf16 cast at the wire boundary). Run with:

    docker run --rm -v "$PWD/packages:/usr/local/app/packages:ro" \
        -v "$PWD/models:/usr/local/app/models:ro" \
        -v "$PWD/scripts/cpu_expert_smoke.py:/tmp/smoke.py:ro" \
        --entrypoint python mixup:latest /tmp/smoke.py
"""

import torch
from safetensors import safe_open
from transformers import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts

from mixup_core.constants import MODEL_DIR, MODEL_FILE
from mixup_core.runtime import RoutingEntry, RoutingTable
from mixup_node.model import ExpertGroup

assert not torch.cuda.is_available(), "this test must run without a GPU"

config = GraniteMoeConfig.from_pretrained(MODEL_DIR)
prefix = "model.layers.0.block_sparse_moe"

# Reference: the full 32 experts, fp32, default (loop) forward.
with safe_open(MODEL_FILE, framework="pt") as f:
    gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:].to(torch.float32)
    down = f.get_slice(f"{prefix}.output_linear.weight")[:].to(torch.float32)
ref = GraniteMoeExperts(config).to("cpu").eval()
ref.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)

# Distributed expert group: a full 32-expert table where this node owns 0..7.
assignment = {
    e: RoutingEntry(node_id="n" if e < 8 else "other", slot=e if e < 8 else e - 8)
    for e in range(32)
}
group = ExpertGroup("n", config, MODEL_FILE, RoutingTable(epoch=0, assignment=assignment))

torch.manual_seed(0)
hidden_bf16 = torch.randn(1, 1024, dtype=torch.bfloat16)
top_k = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
weights = torch.rand(1, 8, dtype=torch.bfloat16)

out_group = group.forward(0, hidden_bf16, top_k, weights)
out_ref = ref(hidden_bf16.to(torch.float32), top_k, weights.to(torch.float32)).to(torch.bfloat16)

diff = (out_group - out_ref).abs().max().item()
print(f"group: device={out_group.device} dtype={out_group.dtype} shape={tuple(out_group.shape)}")
print(f"finite={torch.isfinite(out_group).all().item()} max_abs_diff_vs_ref={diff:.6f}")

assert out_group.shape == (1, 1024)
assert torch.isfinite(out_group).all()
assert diff < 1e-2, f"CPU expert path diverged from reference: {diff}"
print("CPU expert path OK")
