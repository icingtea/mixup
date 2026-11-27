"""Serialize/deserialize the transformers ``DynamicCache`` for network transfer.

The leader snapshots its KV cache every few tokens and broadcasts it so followers hold a
warm copy; a promoted leader deserializes it and resumes without replay. Tensors travel as
safetensors bytes (the same format the expert data plane uses).
"""

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save
from transformers.cache_utils import DynamicCache


def cache_to_bytes(cache: DynamicCache) -> bytes:
    """Serialize a ``DynamicCache``'s per-layer key/value tensors to safetensors bytes."""
    tensors = {}
    for i, layer in enumerate(cache.layers):
        tensors[f"k{i}"] = layer.keys.detach().cpu().contiguous()
        tensors[f"v{i}"] = layer.values.detach().cpu().contiguous()
    return st_save(tensors)


def cache_from_bytes(body: bytes, device: torch.device) -> DynamicCache:
    """Deserialize a ``DynamicCache`` produced by :func:`cache_to_bytes` onto ``device``."""
    tensors = st_load(body)
    n = len(tensors) // 2
    ddp = [(tensors[f"k{i}"].to(device), tensors[f"v{i}"].to(device)) for i in range(n)]
    return DynamicCache(ddp_cache_data=ddp)
