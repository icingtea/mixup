"""GraniteMoE model construction: the central model and per-container expert groups.

The central container holds everything except the experts (embeddings, attention,
routers, final norm); each expert container holds a few experts for every layer.
"""

import copy
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

from mixup_core.topology import experts_for
from mixup_core.types import ExpertRoute

Dispatch = Callable[[int, Tensor, Tensor, Tensor], Tensor]
"""(layer, hidden_states, top_k_index, top_k_weights) -> expert output."""

SKIP = 999
"""Slot number for experts held by another container. transformers skips any slot
number past the end of a container's experts, so these picks do nothing there."""


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
    return model.eval()


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
    """One expert container: holds only its own experts, for every layer."""

    def __init__(
        self,
        name: str,
        model_config: GraniteMoeConfig,
        model_file: Path,
        lut: dict[int, ExpertRoute],
    ) -> None:
        self.name = name
        self.model_config = model_config
        self.lut = lut
        self.num_layers = model_config.num_hidden_layers
        self.num_experts = model_config.num_local_experts or 0
        self.experts = experts_for(lut, name)

        with safe_open(model_file, framework="pt") as f:
            self.layers: list[GraniteMoeExperts] = [
                self._load_layer(f, layer) for layer in range(self.num_layers)
            ]

    def _load_layer(self, file: safe_open, layer: int) -> GraniteMoeExperts:
        """Read this container's experts for one layer out of the big file."""
        prefix = f"model.layers.{layer}.block_sparse_moe"
        all_gate_up = file.get_slice(f"{prefix}.input_linear.weight")
        all_down = file.get_slice(f"{prefix}.output_linear.weight")

        mine = slice(self.experts[0], self.experts[-1] + 1)
        experts = empty_experts(self.model_config, len(self.experts))
        experts.load_state_dict(
            {"gate_up_proj": all_gate_up[mine], "down_proj": all_down[mine]},
            assign=True,
        )
        return experts.eval()

    @torch.no_grad()
    def forward(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """Run this container's share of one layer's experts, zeroing experts it doesn't hold."""
        device = top_k_index.device
        mine = torch.tensor(
            [self.lut[e][0] == self.name for e in range(self.num_experts)], device=device
        )[top_k_index]
        slot = torch.tensor(
            [self.lut[e][1] for e in range(self.num_experts)], device=device
        )[top_k_index]
        slots = torch.where(mine, slot, SKIP)
        weights = top_k_weights.masked_fill(~mine, 0)
        return self.layers[layer](hidden_states, slots, weights)
