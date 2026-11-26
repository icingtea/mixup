"""Tests for the shared logging config and benchmark response fields."""

from mixup_core.logging_config import LOG_FORMAT
from mixup_core.types import GenerateResponse


def test_log_format_is_timestamped() -> None:
    assert "%(asctime)s" in LOG_FORMAT
    assert "%(levelname)s" in LOG_FORMAT
    assert "%(name)s" in LOG_FORMAT


def test_generate_response_carries_benchmark() -> None:
    resp = GenerateResponse(text="hello", tokens=3, elapsed_seconds=0.5)
    assert resp.text == "hello"
    assert resp.tokens == 3
    assert resp.elapsed_seconds == 0.5
