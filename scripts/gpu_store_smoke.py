"""GPU store vs reference (bf16) — run inside a GPU container."""

import torch
from safetensors import safe_open
from transformers import GraniteMoeConfig

from mixup_core.constants import MODEL_DIR, MODEL_FILE
from mixup_core.devices import DeviceProfile
from mixup_core.runtime import RoutingEntry, RoutingTable
from mixup_node.memory import TieredExpertStore
from mixup_node.model import ExpertGroup, empty_experts

assert torch.cuda.is_available(), "needs GPU"
config = GraniteMoeConfig.from_pretrained(MODEL_DIR)
prefix = "model.layers.0.block_sparse_moe"
owned = [0, 8, 13, 16, 19, 21, 24, 31]

# Reference: full 32 experts, grouped_mm (same path ExpertGroup uses).
with safe_open(MODEL_FILE, framework="pt") as f:
    gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:]
    down = f.get_slice(f"{prefix}.output_linear.weight")[:]
ref = empty_experts(config, 32)
ref.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)
ref = ref.to("cuda").eval()

# Tiered store owning `owned`.
profile = DeviceProfile(
    node_id="n", execution="gpu", disk_read_mbps=1000.0, gpu_cpu_bw_mbps=10000.0
)
store = TieredExpertStore("n", config, MODEL_FILE, profile)
store.set_ownership(owned)

# ExpertGroup owning `owned` (grouped_mm path).
assignment = {
    e: RoutingEntry(
        node_id="n" if e in owned else "other",
        slot=owned.index(e) if e in owned else 0,
    )
    for e in range(32)
}
group = ExpertGroup("n", config, MODEL_FILE, RoutingTable(epoch=0, assignment=assignment))

torch.manual_seed(0)
hidden = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
top_k = torch.tensor([[0, 8, 13, 16, 19, 21, 24, 31]], dtype=torch.int64, device="cuda")
weights = torch.rand(1, 8, dtype=torch.bfloat16, device="cuda")

out_store = store.forward(0, hidden, top_k, weights)
out_group = group.forward(0, hidden, top_k, weights)
out_ref = ref(hidden, top_k, weights)

d_sg = (out_store - out_group).abs().max().item()
d_sr = (out_store - out_ref).abs().max().item()
d_gr = (out_group - out_ref).abs().max().item()
print(f"store vs group: {d_sg:.6f}  store vs ref: {d_sr:.6f}  group vs ref: {d_gr:.6f}")
print("store finite:", torch.isfinite(out_store).all().item(), "shape", tuple(out_store.shape))
