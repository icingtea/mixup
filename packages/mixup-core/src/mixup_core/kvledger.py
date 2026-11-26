"""The KV-cache ledger header: schema and codec (tensors live in mixup-node).

The leader is the authoritative writer of an append-only, sequence-numbered KV-cache
log; followers replicate the freshest entry. This module defines only the *header* — the
metadata that travels in gossip and lets a successor rehydrate the cache — not the tensor
bytes (which are safetensors, handled in mixup-node).

References: [Raft] (log replication), [HF-generation].
"""

from pydantic import BaseModel

from .types import Epoch


class LedgerHeader(BaseModel):
    """Metadata for one KV-cache ledger entry (one generated token)."""

    kv_seq: int
    """Monotonic sequence number; the freshest header wins elections."""

    epoch: Epoch
    """The leadership epoch under which this entry was written."""

    input_ids: list[int]
    """Prompt + generated tokens so far (lets a successor audit/fall back to replay)."""

    position_ids: list[int]
    cache_position: list[int]
    attention_mask: list[int]
    layer_shapes: list[tuple[int, int]]
    """(num_kv_heads, head_dim) per layer, to rehydrate ``past_key_values`` correctly."""


def encode_header(header: LedgerHeader) -> bytes:
    """Serialize a header to JSON bytes (the cheap part that travels in gossip)."""
    return header.model_dump_json().encode()


def decode_header(body: bytes) -> LedgerHeader:
    """Deserialize a header from JSON bytes."""
    return LedgerHeader.model_validate_json(body)
