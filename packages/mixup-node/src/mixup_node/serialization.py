"""The wire format between the central node and expert nodes.

Tensors are carried as safetensors bytes; the request and response are pydantic
models so their shapes are explicit and validated at both ends.
"""

from pydantic import BaseModel, ConfigDict
from safetensors.torch import load, save
from torch import Tensor


class ExpertRequest(BaseModel):
    """A POST body to an expert node: run these tokens through one layer's experts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_states: Tensor   # [tokens, hidden] bfloat16
    top_k_index: Tensor     # [tokens, top_k] int64
    top_k_weights: Tensor   # [tokens, top_k] bfloat16

    def to_bytes(self) -> bytes:
        """Serialize to the safetensors bytes sent over HTTP."""
        return save(
            {
                "hidden_states": self.hidden_states.contiguous(),
                "top_k_index": self.top_k_index.contiguous(),
                "top_k_weights": self.top_k_weights.contiguous(),
            }
        )

    @classmethod
    def from_bytes(cls, body: bytes) -> "ExpertRequest":
        """Deserialize from the safetensors bytes received over HTTP."""
        return cls(**load(body))


class ExpertResponse(BaseModel):
    """An expert node's reply: its share of the expert output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    out: Tensor   # [tokens, hidden] bfloat16

    def to_bytes(self) -> bytes:
        return save({"out": self.out})

    @classmethod
    def from_bytes(cls, body: bytes) -> "ExpertResponse":
        return cls(out=load(body)["out"])
