"""Tests for the KV-cache ledger header codec."""

from mixup_core.kvledger import (
    Checkpoint,
    LedgerHeader,
    decode_checkpoint,
    decode_header,
    encode_checkpoint,
    encode_header,
)


def test_header_roundtrip() -> None:
    header = LedgerHeader(
        kv_seq=7,
        epoch=3,
        input_ids=[1, 2, 3],
        position_ids=[0, 1, 2],
        cache_position=[0, 1, 2],
        attention_mask=[1, 1, 1],
        layer_shapes=[(8, 64), (8, 64)],
    )
    assert decode_header(encode_header(header)) == header


def test_checkpoint_roundtrip() -> None:
    checkpoint = Checkpoint(kv_seq=7, input_ids=[1, 2, 3, 4])
    assert decode_checkpoint(encode_checkpoint(checkpoint)) == checkpoint
