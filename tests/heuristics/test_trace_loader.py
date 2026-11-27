"""Tests for loading a real mixup-trace JSONL file."""

from pathlib import Path

from mixup_core.stats import load_trace_file

TRACE = (
    Path(__file__).parent.parent.parent
    / "eval"
    / "traces"
    / "granite-1b-instruct-100tok.jsonl"
)


def test_load_real_trace_shape() -> None:
    stats = load_trace_file(TRACE)
    assert stats.num_layers == 24
    assert stats.num_experts == 32
    assert stats.total_tokens == 100
    # Top-8 routing: every layer sums to 8 selections per token.
    for layer in stats.marginals():
        assert abs(sum(layer) - 8.0) < 1e-9


def test_load_real_trace_top_experts_are_in_range() -> None:
    stats = load_trace_file(TRACE)
    for layer in range(stats.num_layers):
        assert all(0 <= e < 32 for e in stats.top_experts(layer, 8))


def test_load_trace_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    import pytest

    with pytest.raises(ValueError):
        load_trace_file(empty)
