"""Store vs reference across layers, with batched (prefill-like) hidden states."""

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


def table_for(name, owned):
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

torch.manual_seed(3)
for layer in (0, 5, 11, 23):
    prefix = f"model.layers.{layer}.block_sparse_moe"
    with safe_open(MODEL_FILE, framework="pt") as f:
        gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:]
        down = f.get_slice(f"{prefix}.output_linear.weight")[:]
    ref = empty_experts(config, 32)
    ref.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)
    ref = ref.to("cuda").eval()

    # Batched (7 tokens, prefill-like).
    hidden = torch.randn(7, 1024, dtype=torch.bfloat16, device="cuda")
    top_k = torch.randint(0, 32, (7, 8), dtype=torch.int64, device="cuda")
    w = torch.rand(7, 8, dtype=torch.bfloat16, device="cuda")

    o1 = store.forward(layer, hidden, top_k, w)
    o2 = g2.forward(layer, hidden, top_k, w)
    o3 = g3.forward(layer, hidden, top_k, w)
    total = o1 + o2 + o3
    oref = ref(hidden, top_k, w)
    diff = (total - oref).abs().max().item()
    print(
        f"layer {layer}: total={diff:.6f} finite={torch.isfinite(total).all().item()}"
    )
    # The distributed sum matches within bf16 partial-sum rounding.
    assert diff < 0.1, diff
print("batched store vs ref OK")
