"""TieredExpertStore CPU smoke test (run inside a CPU-only container).

Verifies the per-expert store computes the same output as a direct fp32 reference, and
that a byte budget forces eviction (resident count stays under the cap).
"""

import torch
from safetensors import safe_open
from transformers import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts

from mixup_core.constants import MODEL_DIR, MODEL_FILE
from mixup_core.devices import DeviceProfile
from mixup_node.memory import TieredExpertStore

assert not torch.cuda.is_available(), "this test must run without a GPU"

config = GraniteMoeConfig.from_pretrained(MODEL_DIR)
prefix = "model.layers.0.block_sparse_moe"
owned = list(range(8))

# Reference: full 32 experts, fp32, default loop forward.
with safe_open(MODEL_FILE, framework="pt") as f:
    gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:].to(torch.float32)
    down = f.get_slice(f"{prefix}.output_linear.weight")[:].to(torch.float32)
ref = GraniteMoeExperts(config).to("cpu").eval()
ref.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)

profile = DeviceProfile(node_id="n", execution="cpu", disk_read_mbps=500.0)
store = TieredExpertStore("n", config, MODEL_FILE, profile)
store.set_ownership(owned)

torch.manual_seed(0)
hidden = torch.randn(1, 1024, dtype=torch.bfloat16)
top_k = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
weights = torch.rand(1, 8, dtype=torch.bfloat16)

out = store.forward(0, hidden, top_k, weights)
out_ref = ref(hidden.to(torch.float32), top_k, weights.to(torch.float32)).to(torch.bfloat16)
diff = (out - out_ref).abs().max().item()
finite = torch.isfinite(out).all()
print(
    f"forward: shape={tuple(out.shape)} dtype={out.dtype} finite={finite} max_diff={diff:.6f}"
)
assert diff < 1e-2, diff

# Budget: cap the store at 3 resident experts; ensure evicts the rest.
budget = store._expert_bytes() * 3
store2 = TieredExpertStore("n", config, MODEL_FILE, profile, budget_bytes={"cpu": budget})
store2.set_ownership(owned)
for e in owned:
    store2.ensure(0, e)
print(f"budget {budget} bytes -> resident {store2.resident_experts()} experts")
assert store2.resident_experts() <= 3
assert store2.residency.used_bytes("cpu") <= budget
print("TieredExpertStore OK")
