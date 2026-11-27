"""The tiered expert store: per-(layer, expert) weights with GPU/CPU/disk residency.

This is the node-side half of the offloading layer. The pure ``Residency`` core decides
*what* to keep/evict/prefetch; this class applies those decisions to real tensors:

- an expert's weights live on the execution tier (GPU or CPU) or on disk (the model file);
- ``ensure`` loads from disk (or moves from CPU RAM) on demand, evicting under budget;
- ``forward`` computes the node's share of one layer by running each owned selected expert
  individually (matching the model's own per-expert loop), loading on miss.

The wire is bf16; CPU nodes compute in fp32 and cast back (see ``model._expert_dtype``).
"""

import threading
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F
from transformers import GraniteMoeConfig

from mixup_core.devices import DeviceProfile
from mixup_core.residency import Residency, prefetch_threshold
from mixup_core.tiers import Tier

from .model import _expert_dtype


class TieredExpertStore:
    """Owns a set of experts (all layers) and manages their tier residency."""

    def __init__(
        self,
        name: str,
        model_config: GraniteMoeConfig,
        model_file: Path,
        profile: DeviceProfile,
        budget_bytes: int | None = None,
    ) -> None:
        self.name = name
        self.model_config = model_config
        self.model_file = model_file
        self.num_layers = model_config.num_hidden_layers
        self.num_experts = model_config.num_local_experts or 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _expert_dtype(self.device)
        self.execution_tier = Tier.GPU if self.device.type == "cuda" else Tier.CPU

        self.owned: set[int] = set()
        # (layer, expert) -> {gate_up, down} on the execution tier.
        self._weights: dict[tuple[int, int], dict[str, Tensor]] = {}
        # (layer, expert) -> predicted next-activation probability (from the leader's
        # prefetch hint). Drives the eviction score's ``(1 - δ)(1 - p)`` term and the
        # ``θ_pre`` prefetch threshold (Phase 3 / CoMoE fine-grained residency).
        self._predicted: dict[tuple[int, int], float] = {}
        self._last_prefetch_threshold: float | None = None  # for the observability UI
        budget = {self.execution_tier: budget_bytes} if budget_bytes is not None else {}
        self.residency = Residency(
            profile=profile,
            expert_bytes_per_layer=self._expert_bytes(),
            budget_bytes=budget,
        )
        self._file = None  # opened lazily for disk loads

    @property
    def experts(self) -> list[int]:
        """The owned expert ids, sorted (mirrors ``ExpertGroup.experts``)."""
        return sorted(self.owned)

    def _expert_bytes(self) -> int:
        hidden = self.model_config.hidden_size
        inter = self.model_config.intermediate_size
        # gate_up (2*inter*hidden) + down (hidden*inter) parameters.
        params = 2 * inter * hidden + hidden * inter
        return params * (4 if self.dtype == torch.float32 else 2)

    def set_ownership(self, experts: list[int]) -> None:
        """Adopt ``experts`` (global ids); drop weights for experts no longer owned."""
        new = set(experts)
        for key in list(self._weights):
            if key[1] not in new:
                self._drop(key)
        self.owned = new

    def set_routing(self, table) -> None:
        """Adopt a routing table (drop-in for ``ExpertGroup.set_routing``).

        Extracts this node's owned experts from the table and reloads only the changed set.
        """
        from mixup_core.runtime import RoutingTable

        if not isinstance(table, RoutingTable):
            table = RoutingTable.model_validate(table)
        owned = sorted(
            e for e, entry in table.assignment.items() if entry.node_id == self.name
        )
        if owned == sorted(self.owned):
            return
        self.set_ownership(owned)

    def _open(self) -> safe_open:
        if self._file is None:
            self._file = safe_open(self.model_file, framework="pt")
        return self._file

    def _load_from_disk(self, layer: int, expert: int) -> dict[str, Tensor]:
        prefix = f"model.layers.{layer}.block_sparse_moe"
        f = self._open()
        gate_up = f.get_slice(f"{prefix}.input_linear.weight")[expert : expert + 1]
        down = f.get_slice(f"{prefix}.output_linear.weight")[expert : expert + 1]
        gate_up = gate_up.reshape(gate_up.shape[-2], gate_up.shape[-1]).to(self.device, self.dtype)
        down = down.reshape(down.shape[-2], down.shape[-1]).to(self.device, self.dtype)
        return {"gate_up": gate_up, "down": down}

    def _drop(self, key: tuple[int, int]) -> None:
        self._weights.pop(key, None)
        self.residency.place(key[0], key[1], Tier.DISK, 0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _evict_one(self) -> None:
        """Evict the lowest-retention resident expert (moving it to disk).

        The eviction order uses the paper's ``S_evict``, which down-weights experts the
        predictor expects to fire again soon (``(1 - δ)(1 - p)``), then breaks ties on
        recency and importance.
        """
        order = self.residency.eviction_order(self.execution_tier, self._predicted)
        for key in order:
            if key in self._weights:
                self._drop(key)
                return

    def ensure(self, layer: int, expert: int) -> dict[str, Tensor]:
        """Return the weights for ``(layer, expert)`` on the execution tier, loading on miss."""
        key = (layer, expert)
        if key in self._weights:
            return self._weights[key]
        weights = self._load_from_disk(layer, expert)
        self._weights[key] = weights
        self.residency.place(layer, expert, self.execution_tier, self._expert_bytes())
        self.residency.touch(layer, expert)
        # Enforce the execution-tier budget: evict until the new expert fits.
        budget = self.residency.budget_bytes.get(self.execution_tier)
        if budget is not None:
            while self.residency.used_bytes(self.execution_tier) > budget:
                self._evict_one()
        return weights

    @torch.no_grad()
    def forward(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """Run this node's share of one layer: owned selected experts, loading on miss.

        Handles batched inputs (the prefill sends all prompt tokens at once); each token
        contributes its own weighted experts.
        """
        if not self.owned:
            return torch.zeros_like(hidden_states)
        device = self.device
        hidden = hidden_states.to(device, self.dtype)  # (T, hidden)
        top_k_index = top_k_index.to(device)  # (T, K) int64
        top_k_weights = top_k_weights.to(device, self.dtype)  # (T, K)

        # Accumulate in fp32 (like the model's grouped_mm path, which sums in fp32 to
        # avoid bf16 accumulation error); cast back to bf16 only at the wire boundary.
        out = torch.zeros(hidden.shape, dtype=torch.float32, device=device)
        num_tokens, top_k = top_k_index.shape
        for token in range(num_tokens):
            for pos in range(top_k):
                expert = int(top_k_index[token, pos])
                if expert not in self.owned:
                    continue
                w = self.ensure(layer, expert)
                gate, up = F.linear(hidden[token], w["gate_up"]).chunk(2, dim=-1)
                y = F.silu(gate) * up
                y = F.linear(y, w["down"])
                out[token] += (y * top_k_weights[token, pos]).to(torch.float32)
        return out.to(torch.bfloat16)

    def resident_experts(self) -> int:
        return len(self._weights)

    def residency_info(self) -> dict:
        """Expose residency state for the observability UI: the eviction budget, the
        used/free bytes on the execution tier, and the eviction/prefetch formula tunables."""
        tier = self.execution_tier
        cfg = self.residency.config
        budget = self.residency.budget_bytes.get(tier)
        return {
            "execution_tier": tier,
            "budget_bytes": budget,
            "used_bytes": self.residency.used_bytes(tier),
            "free_bytes": self.residency.free_bytes(tier),
            "free_ratio": round(self.residency.free_ratio(tier), 4),
            "resident_count": len(self._weights),
            "prefetch_threshold": (
                round(self._last_prefetch_threshold, 4)
                if self._last_prefetch_threshold is not None
                else None
            ),
            "config": {
                "gamma": cfg.gamma,
                "delta": cfg.delta,
                "lambda": cfg.lambd,
                "theta_base": cfg.theta_base,
                "theta_mem": cfg.theta_mem,
                "decay": cfg.decay,
            },
        }

    def prefetch(
        self,
        layer: int,
        experts: list[int],
        probs: list[float] | None = None,
        stability: float = 1.0,
    ) -> None:
        """Prefetch ``experts`` for ``layer`` in the background (Phase 3 + 6).

        Applies the paper's ``θ_pre`` threshold: only experts whose predicted activation
        probability clears the threshold are prefetched, and the threshold itself lowers as
        memory frees up or the resource state stabilizes. A miss falls back to the blocking
        load in ``ensure``, so correctness never depends on the hint.
        """
        if not experts:
            return
        probs = list(probs) if probs is not None else [1.0] * len(experts)

        # Record the predicted probabilities so eviction can down-weight soon-to-be-used
        # experts (the ``(1 - δ)(1 - p)`` term) instead of treating every expert as 0.
        for e, p in zip(experts, probs, strict=False):
            if e in self.owned:
                self._predicted[(layer, e)] = float(p)

        cfg = self.residency.config
        threshold = prefetch_threshold(
            cfg.theta_base,
            stability,
            self.residency.free_ratio(self.execution_tier),
            cfg.theta_mem,
        )
        self._last_prefetch_threshold = threshold
        to_load = [
            e for e, p in zip(experts, probs, strict=False) if e in self.owned and p >= threshold
        ]
        if not to_load:
            return

        def _load() -> None:
            for expert in to_load:
                self.ensure(layer, expert)

        threading.Thread(target=_load, daemon=True).start()
