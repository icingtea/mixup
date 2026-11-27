"""GraniteMoE model construction: the central model and per-container expert groups.

The central container holds everything except the experts (embeddings, attention,
routers, final norm); each expert container holds a few experts for every layer.
"""

import copy
import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import Tensor
from transformers import AutoModelForCausalLM, GraniteMoeConfig, GraniteMoeForCausalLM
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeDecoderLayer,
    GraniteMoeExperts,
)

from mixup_core.runtime import RoutingTable
from mixup_core.topology import experts_for
from mixup_core.types import ExpertRoute

Dispatch = Callable[[int, Tensor, Tensor, Tensor], Tensor]
"""(layer, hidden_states, top_k_index, top_k_weights) -> expert output."""

SKIP = 999
"""Slot number for experts held by another container. transformers skips any slot
number past the end of a container's experts, so these picks do nothing there."""

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    """The compute device: cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expert_dtype(device: torch.device) -> torch.dtype:
    """The dtype experts compute in on this device: bf16 on GPU, fp32 on CPU.

    CPU (especially ARM Pis) has weak bf16 support, so CPU expert nodes compute in fp32
    and cast their output back to bf16 at the wire boundary to match the leader.
    """
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def empty_model(model_config: GraniteMoeConfig) -> GraniteMoeForCausalLM:
    """Build the whole model with no weights yet (meta device: shapes but no data)."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config, dtype=torch.bfloat16)
    return cast(GraniteMoeForCausalLM, model)


class ExpertSlot(torch.nn.Module):
    """Sits where a layer's experts were, on the central container.

    transformers calls it after the router has picked experts; it hands the work to
    ``dispatch``, which runs the experts on the containers that hold them.
    """

    def __init__(self, layer: int, dispatch: Dispatch) -> None:
        super().__init__()
        self.layer = layer
        self.dispatch = dispatch

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        return self.dispatch(self.layer, hidden_states, top_k_index, top_k_weights)


def load_central(
    model_config: GraniteMoeConfig,
    model_file: Path,
    dispatch: Dispatch,
) -> GraniteMoeForCausalLM:
    """The central container's model: everything except the experts, ready to generate."""
    model = empty_model(model_config)
    layers = cast(list[GraniteMoeDecoderLayer], model.model.layers)
    for index, layer in enumerate(layers):
        layer.block_sparse_moe.experts = ExpertSlot(index, dispatch)

    with safe_open(model_file, framework="pt") as f:
        state: dict[str, Tensor] = {
            key.replace("router.layer.weight", "router.weight"): f.get_tensor(key)
            for key in f.keys()  # noqa: SIM118 - safe_open is not a mapping
            if "input_linear" not in key and "output_linear" not in key
        }
    model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()
    model.model.rotary_emb = type(model.model.rotary_emb)(model_config)
    return model.to(_device()).eval()


def empty_experts(model_config: GraniteMoeConfig, num_experts: int) -> GraniteMoeExperts:
    """transformers' experts module sized for ``num_experts`` instead of all of them."""
    small_config = copy.deepcopy(model_config)
    small_config.num_local_experts = num_experts
    small_config._experts_implementation = "grouped_mm"
    with torch.device("meta"):
        experts = GraniteMoeExperts(small_config)
    experts._is_expert_parallel = True
    return experts


class ExpertGroup:
    """One expert container: holds only its own experts, for every layer.

    ``set_routing`` swaps in a new assignment (used on failover): it reloads the experts
    only if this node's set changed, so surviving nodes that keep their experts do no
    work. Loading is per-expert (a union of runs), so a node can own a non-contiguous set
    after a rebalance.
    """

    def __init__(
        self,
        name: str,
        model_config: GraniteMoeConfig,
        model_file: Path,
        table: RoutingTable,
    ) -> None:
        self.name = name
        self.model_config = model_config
        self.model_file = model_file
        self.num_layers = model_config.num_hidden_layers
        self.num_experts = model_config.num_local_experts or 0
        self._lut: dict[int, ExpertRoute] = {}
        self.experts: list[int] = []
        self.layers: list[GraniteMoeExperts] = []
        self._slot_or_skip: Tensor | None = None
        self._owned: Tensor | None = None
        self.set_routing(table)

    def set_routing(self, table: RoutingTable) -> None:
        """Adopt ``table``; reload experts only if this node's set changed."""
        lut = {e: (r.node_id, r.slot) for e, r in table.assignment.items()}
        experts = experts_for(lut, self.name)
        if experts == self.experts:
            return
        added = sorted(set(experts) - set(self.experts))
        removed = sorted(set(self.experts) - set(experts))
        self._lut = lut
        self.experts = experts
        self._slot_or_skip, self._owned = self._mask_tensors()
        # Free the old experts' GPU memory before loading the new set. Repeated
        # rebalances otherwise leave orphaned expert copies cached in the CUDA
        # allocator, which on a shared GPU starves the other containers (OOM /
        # CUBLAS_STATUS_ALLOC_FAILED). empty_cache() returns the blocks to the
        # driver so sibling processes can actually use them.
        self.layers = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.layers = self._load_all() if experts else []
        logger.info(
            "%s: experts now %s (gained %s, lost %s)",
            self.name,
            experts,
            added,
            removed,
        )

    def _mask_tensors(self) -> tuple[Tensor, Tensor]:
        """Precompute per-expert dispatch masks (one device copy each).

        ``slot_or_skip[e]`` is the local slot of an owned expert or ``SKIP``; ``owned[e]``
        is 1.0/0.0. They depend only on the routing table, not on the request, so they
        are built once here and indexed per request. This replaces the previous
        per-call work — building two 32-element masks, plus a ``where`` and a
        ``masked_fill`` — with two gathers, saving several GPU syncs on each of the
        ~4.4k expert requests per generation.
        """
        device = _device()
        slot_or_skip = torch.tensor(
            [
                self._lut[e][1] if self._lut[e][0] == self.name else SKIP
                for e in range(self.num_experts)
            ],
            dtype=torch.int64,
            device=device,
        )
        owned = torch.tensor(
            [1.0 if self._lut[e][0] == self.name else 0.0 for e in range(self.num_experts)],
            dtype=torch.bfloat16,
            device=device,
        )
        return slot_or_skip, owned

    def _load_all(self) -> list[GraniteMoeExperts]:
        with safe_open(self.model_file, framework="pt") as f:
            return [self._load_layer(f, layer) for layer in range(self.num_layers)]

    def _load_layer(self, file: safe_open, layer: int) -> GraniteMoeExperts:
        """Read this node's experts for one layer out of the big file (any set, sorted)."""
        prefix = f"model.layers.{layer}.block_sparse_moe"
        all_gate_up = file.get_slice(f"{prefix}.input_linear.weight")
        all_down = file.get_slice(f"{prefix}.output_linear.weight")

        gate_up = torch.cat([all_gate_up[e : e + 1] for e in self.experts], dim=0)
        down = torch.cat([all_down[e : e + 1] for e in self.experts], dim=0)
        device = _device()
        if device.type == "cpu":
            gate_up = gate_up.to(torch.float32)
            down = down.to(torch.float32)
        experts = empty_experts(self.model_config, len(self.experts))
        experts.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)
        return experts.to(device).eval()

    @torch.no_grad()
    def forward(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """Run this node's share of one layer's experts, zeroing experts it doesn't hold."""
        if not self.experts:
            return torch.zeros_like(hidden_states)
        device = _device()
        dtype = _expert_dtype(device)
        hidden_states = hidden_states.to(device, dtype)
        top_k_index = top_k_index.to(device)
        top_k_weights = top_k_weights.to(device, dtype)
        assert self._slot_or_skip is not None and self._owned is not None
        slots = self._slot_or_skip[top_k_index]
        weights = top_k_weights * self._owned[top_k_index]
        out = self.layers[layer](hidden_states, slots, weights)
        # Return fp32: the leader reduces in fp32 and casts to bf16 once, so the cross-node
        # sum is order-independent (deterministic across a failover that reshuffles experts).
        return out.to(torch.float32)
