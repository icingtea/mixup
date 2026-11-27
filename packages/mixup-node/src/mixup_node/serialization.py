"""The wire format between the central node and expert nodes.

Tensors are carried as safetensors bytes; the request and response are pydantic
models so their shapes are explicit and validated at both ends.
"""

import torch
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load, save
from torch import Tensor


class ExpertRequest(BaseModel):
    """A POST body to an expert node: run these tokens through one layer's experts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_states: Tensor   # [tokens, hidden] bfloat16
    top_k_index: Tensor     # [tokens, top_k] int64
    top_k_weights: Tensor   # [tokens, top_k] bfloat16
    prefetch: Tensor        # [k] int64 — predicted next-layer experts (hint), may be empty
    prefetch_probs: Tensor  # [k] float32 — their predicted activation probability

    def to_bytes(self) -> bytes:
        """Serialize to the safetensors bytes sent over HTTP (moves to CPU)."""
        prefetch = self.prefetch.contiguous().cpu().to(torch.int64)
        probs = self.prefetch_probs.contiguous().cpu().to(torch.float32)
        return save(
            {
                "hidden_states": self.hidden_states.contiguous().cpu(),
                "top_k_index": self.top_k_index.contiguous().cpu(),
                "top_k_weights": self.top_k_weights.contiguous().cpu(),
                "prefetch": prefetch,
                "prefetch_probs": probs,
            }
        )

    @classmethod
    def from_bytes(cls, body: bytes) -> "ExpertRequest":
        """Deserialize from the safetensors bytes received over HTTP."""
        d = load(body)
        if "prefetch" not in d:
            d["prefetch"] = torch.tensor([], dtype=torch.int64)
        if "prefetch_probs" not in d:
            d["prefetch_probs"] = torch.zeros(d["prefetch"].numel(), dtype=torch.float32)
        return cls(**d)


class ExpertResponse(BaseModel):
    """An expert node's reply: its share of the expert output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    out: Tensor   # [tokens, hidden] bfloat16

    def to_bytes(self) -> bytes:
        return save({"out": self.out.contiguous().cpu()})

    @classmethod
    def from_bytes(cls, body: bytes) -> "ExpertResponse":
        return cls(out=load(body)["out"])
