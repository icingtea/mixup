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


class Checkpoint(BaseModel):
    """A token-sequence checkpoint for resume (the plan's replay/recovery path).

    ``input_ids`` is the full sequence so far (prompt + generated); ``kv_seq`` is how many
    tokens have been generated; ``max_new_tokens`` is the target, so a promoted leader can
    compute how many remain without any out-of-band state.
    """

    kv_seq: int
    input_ids: list[int]
    max_new_tokens: int


def encode_checkpoint(checkpoint: Checkpoint) -> bytes:
    """Serialize a checkpoint to JSON bytes."""
    return checkpoint.model_dump_json().encode()


def decode_checkpoint(body: bytes) -> Checkpoint:
    """Deserialize a checkpoint from JSON bytes."""
    return Checkpoint.model_validate_json(body)


class KvState(BaseModel):
    """A warm KV-cache snapshot the leader broadcasts so followers can resume it.

    ``cache_b64`` is the base64 of the safetensors-serialized ``DynamicCache``; the other
    fields are the metadata a successor needs to continue: how many tokens were generated
    (``kv_seq``), the next token id already computed (``next_token``), the full token
    sequence so far (``input_ids`` = prompt + generated), and the generation target.
    """

    kv_seq: int
    next_token: int
    input_ids: list[int]
    max_new_tokens: int
    cache_b64: str


def encode_kv_state(state: KvState) -> bytes:
    """Serialize a KvState to JSON bytes."""
    return state.model_dump_json().encode()


def decode_kv_state(body: bytes) -> KvState:
    """Deserialize a KvState from JSON bytes."""
    return KvState.model_validate_json(body)
