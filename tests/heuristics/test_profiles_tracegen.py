"""Tests for concrete profiles and synthetic trace generation."""

from mixup_core.devices import ExecutionType
from mixup_core.profiles import TESTBED, DeviceSpec, build_testbed_profiles
from mixup_core.tracegen import generate_trace, trace_to_stats


def test_spec_to_profile_roundtrip() -> None:
    spec = DeviceSpec(execution="cpu", cpu_cores=2, expert_per_ms=4.0)
    p = spec.to_profile("n1")
    assert p.node_id == "n1"
    assert p.execution is ExecutionType.CPU
    assert p.cpu_cores == 2
    assert p.expert_per_ms == 4.0


def test_testbed_has_five_devices() -> None:
    assert set(TESTBED) == {"laptop-5060", "laptop-3070", "laptop-3060", "pi-1", "pi-2"}


def test_testbed_profiles_execution_types() -> None:
    profiles = build_testbed_profiles()
    assert profiles["laptop-5060"].is_gpu
    assert not profiles["pi-1"].is_gpu
    assert profiles["pi-1"].vram_bytes is None


def test_generate_trace_shape_and_determinism() -> None:
    trace = generate_trace(num_layers=3, num_experts=8, top_k=4, num_tokens=10, seed=7)
    assert len(trace) == 10
    assert all(len(token) == 3 for token in trace)
    assert all(len(layer) == 4 for token in trace for layer in token)
    assert trace == generate_trace(num_layers=3, num_experts=8, top_k=4, num_tokens=10, seed=7)


def test_generate_trace_experts_in_range() -> None:
    trace = generate_trace(num_layers=2, num_experts=6, top_k=3, num_tokens=5, seed=1)
    for token in trace:
        for layer in token:
            assert all(0 <= e < 6 for e in layer)


def test_trace_to_stats_counts_tokens() -> None:
    trace = generate_trace(num_layers=2, num_experts=6, top_k=3, num_tokens=12, seed=2)
    stats = trace_to_stats(trace, 2, 6)
    assert stats.total_tokens == 12
    # marginals sum to top_k per layer.
    for layer in stats.marginals():
        assert abs(sum(layer) - 3.0) < 1e-9
