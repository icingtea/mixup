"""Full distributed forward vs reference, for a realistic top_k (GPU)."""

import torch
from safetensors import safe_open
from transformers import GraniteMoeConfig

from mixup_core.constants import MODEL_DIR, MODEL_FILE
from mixup_core.devices import DeviceProfile
from mixup_core.runtime import RoutingEntry, RoutingTable
from mixup_node.memory import TieredExpertStore
from mixup_node.model import ExpertGroup, empty_experts

assert torch.cuda.is_available()
config = GraniteMoeConfig.from_pretrained(MODEL_DIR)

c1 = [0, 8, 13, 16, 19, 20, 24, 31]
c2 = [2, 4, 9, 10, 12, 17, 18, 23, 27, 28, 29, 30]
c3 = [1, 3, 5, 6, 7, 11, 14, 15, 21, 22, 25, 26]
assert sorted(c1 + c2 + c3) == list(range(32))


def table_for(name: str, owned: list[int]) -> RoutingTable:
    return RoutingTable(
        epoch=0,
        assignment={
            e: RoutingEntry(
                node_id=name if e in owned else "other",
                slot=owned.index(e) if e in owned else 0,
            )
            for e in range(32)
        },
    )


profile = DeviceProfile(
    node_id="n", execution="gpu", disk_read_mbps=1000.0, gpu_cpu_bw_mbps=10000.0
)
store = TieredExpertStore("n", config, MODEL_FILE, profile)
store.set_ownership(c1)
g2 = ExpertGroup("n2", config, MODEL_FILE, table_for("n2", c2))
g3 = ExpertGroup("n3", config, MODEL_FILE, table_for("n3", c3))

prefix = "model.layers.0.block_sparse_moe"
with safe_open(MODEL_FILE, framework="pt") as f:
    gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:]
    down = f.get_slice(f"{prefix}.output_linear.weight")[:]
ref = empty_experts(config, 32)
ref.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)
ref = ref.to("cuda").eval()

torch.manual_seed(1)
hidden = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
# A realistic top_k (global ids, mix of all three owners).
top_k = torch.tensor([[0, 1, 8, 13, 17, 20, 24, 31]], dtype=torch.int64, device="cuda")
w = torch.rand(1, 8, dtype=torch.bfloat16, device="cuda")

o1 = store.forward(0, hidden, top_k, w)
o2 = g2.forward(0, hidden, top_k, w)
o3 = g3.forward(0, hidden, top_k, w)
total = o1 + o2 + o3
oref = ref(hidden, top_k, w)

finite = f"store={torch.isfinite(o1).all()} g2={torch.isfinite(o2).all()} g3={torch.isfinite(o3).all()}"
print(finite)
print(f"max diff (distributed vs ref): {(total - oref).abs().max().item():.6f}")

# Per-node reference partials: run ref on each node's selected subset.
def ref_partial(owned: list[int]) -> torch.Tensor:
    idxs = [i for i, e in enumerate(top_k.flatten().tolist()) if e in owned]
    if not idxs:
        return torch.zeros_like(oref)
    sub_k = top_k[:, idxs]
    sub_w = w[:, idxs]
    return ref(hidden, sub_k, sub_w)


r1 = ref_partial(c1)
r2 = ref_partial(c2)
r3 = ref_partial(c3)
print(f"store vs ref(c1): {(o1 - r1).abs().max().item():.6f}")
print(f"g2     vs ref(c2): {(o2 - r2).abs().max().item():.6f}")
print(f"g3     vs ref(c3): {(o3 - r3).abs().max().item():.6f}")
