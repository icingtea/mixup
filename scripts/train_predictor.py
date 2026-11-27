"""Train + evaluate the LR and MLP predictors on real generation traces (run in a container).

Captures per-layer hidden states + top-k during greedy generation, builds ``(features,
label)`` samples where features = ``[h_l, one_hot(e_l), 1]`` and the label is the actual
next-layer top-k, then trains LR and MLP online and reports top-1/top-3 vs the frequency
baseline. This is the Phase 6 go/no-go measurement.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixup_core.constants import MODEL_DIR
from mixup_core.stats import ActivationStats
from mixup_node.predictor import LRPredictor, MLPPredictor, _labels, features


def _hook_layers(model, num_layers: int):
    hidden: dict[int, torch.Tensor] = {}
    topk: dict[int, list[int]] = {}
    handles = []

    def make_hook(layer: int):
        def hook(module, args, output):
            hidden[layer] = args[0][-1].detach().float()  # last token's hidden state
            topk[layer] = output[2][-1].tolist()  # (logits, weights, index)

        return hook

    for layer, dec_layer in enumerate(model.model.layers[:num_layers]):
        handles.append(dec_layer.block_sparse_moe.router.register_forward_hook(make_hook(layer)))
    return hidden, topk, handles


def _collect(prompt: str, max_new_tokens: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    hidden, topk, handles = _hook_layers(model, num_layers)

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    stats = ActivationStats(num_layers=num_layers, num_experts=num_experts)
    try:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=prompt_ids, cache=None, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1)
            for _ in range(max_new_tokens):
                if nxt.item() == tokenizer.eos_token_id:
                    break
                out = model(input_ids=nxt.unsqueeze(0), cache=out.cache, use_cache=True)
                layers = [topk[ly] for ly in range(num_layers)]
                stats.record(layers)
                for ly in range(num_layers - 1):
                    x = features(hidden[ly], topk[ly], num_experts)
                    y = _labels(topk[ly + 1], num_experts)
                    xs.append(x)
                    ys.append(y)
                nxt = out.logits[:, -1, :].argmax(dim=-1)
    finally:
        for h in handles:
            h.remove()
    return xs, ys, stats, num_experts


def _topk_accuracy(preds: torch.Tensor, ys: torch.Tensor, k: int) -> float:
    """Fraction of rows where one of the top-k predicted experts is a true label."""
    top = preds.topk(k, dim=1).indices.cpu()
    hits = 0
    for i in range(preds.shape[0]):
        row = ys[i].cpu() if ys[i].device.type == "cuda" else ys[i]
        if any(float(row[t]) > 0 for t in top[i]):
            hits += 1
    return hits / preds.shape[0]


def main() -> None:
    print("collecting training samples...")
    xs, ys, stats, num_experts = _collect("Explain mixture of experts models in detail.", 120)
    print(f"{len(xs)} samples, {num_experts} experts")

    # Split: train on first 70%, eval on last 30%.
    split = int(len(xs) * 0.7)
    feat_dim = xs[0].shape[0]
    lr = LRPredictor(feat_dim, num_experts)
    mlp = MLPPredictor(feat_dim, num_experts, hidden_dim=64)

    print("training LR + MLP online...")
    for i in range(split):
        x, y = xs[i], ys[i]
        lr.update(x, y)
        mlp.update(x, y)

    # Evaluate.
    X = torch.stack([xs[i] for i in range(split, len(xs))])
    Y = torch.stack([ys[i] for i in range(split, len(xs))])
    lr_preds = torch.sigmoid(torch.stack([lr.predict(x) for x in X]))
    mlp_preds = torch.sigmoid(torch.stack([mlp.predict(x) for x in X]))

    print(
        f"LR  top-1={_topk_accuracy(lr_preds, Y, 1):.3f} top-3={_topk_accuracy(lr_preds, Y, 3):.3f}"
    )
    print(
        f"MLP top-1={_topk_accuracy(mlp_preds, Y, 1):.3f} "
        f"top-3={_topk_accuracy(mlp_preds, Y, 3):.3f}"
    )

    # Frequency baseline on the eval set (using the layer-agnostic hot set).
    freq_top = torch.tensor(
        sorted(
            range(num_experts),
            key=lambda e: -sum(stats.marginals()[ly][e] for ly in range(stats.num_layers)),
        )
    )
    freq_preds = torch.zeros(Y.shape)
    for i in range(Y.shape[0]):
        freq_preds[i] = torch.nn.functional.one_hot(freq_top, num_experts).sum(0).float()
    print(
        f"freq top-1={_topk_accuracy(freq_preds, Y, 1):.3f} "
        f"top-3={_topk_accuracy(freq_preds, Y, 3):.3f}"
    )


if __name__ == "__main__":
    main()
