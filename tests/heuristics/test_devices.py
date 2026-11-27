"""Tests for device profiles, resource smoothing, and activation statistics."""

import pytest

from mixup_core.devices import DeviceProfile, ExecutionType, ResourceState
from mixup_core.resource import (
    MetricTracker,
    ewma,
    heterogeneity,
    normalized_difference,
    stability,
)
from mixup_core.stats import ActivationStats, TraceBuffer

# -- DeviceProfile -------------------------------------------------------


def test_execution_type_is_str_enum() -> None:
    assert ExecutionType.GPU.value == "gpu"
    assert ExecutionType.CPU.value == "cpu"
    assert ExecutionType("gpu") is ExecutionType.GPU


def test_profile_is_gpu() -> None:
    assert DeviceProfile(node_id="n1", execution="gpu").is_gpu
    assert not DeviceProfile(node_id="n2", execution="cpu").is_gpu


def test_profile_defaults() -> None:
    p = DeviceProfile(node_id="n1", execution="cpu")
    assert p.vram_bytes is None
    assert p.gpu_tflops is None
    assert p.can_lead is True
    assert p.lead_score == 0


def test_resource_state_defaults() -> None:
    r = ResourceState()
    assert r.cpu_util == 0.0
    assert r.free_vram_bytes is None
    assert r.rtt_ms == {}


# -- EWMA / stability ----------------------------------------------------


def test_ewma_converges_to_constant() -> None:
    v = 0.0
    for _ in range(100):
        v = ewma(v, 5.0, 0.3)
    assert v == pytest.approx(5.0, rel=1e-6)


def test_ewma_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        ewma(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        ewma(0.0, 1.0, 1.5)


def test_ewma_weighting() -> None:
    # First step: alpha*value + (1-alpha)*prev.
    assert ewma(10.0, 20.0, 0.5) == pytest.approx(15.0)


def test_stability_perfect_series_is_one() -> None:
    assert stability([5.0, 5.0, 5.0], 5.0) == pytest.approx(1.0)


def test_stability_volatile_series_is_low() -> None:
    s = stability([0.0, 100.0, 0.0, 100.0], 50.0)
    assert s < 0.5


def test_stability_zero_smoothed_is_zero() -> None:
    assert stability([0.0, 0.0], 0.0) == 0.0


def test_stability_is_clipped_to_unit() -> None:
    # Negative volatility contribution clamps to >= 0.
    assert stability([100.0, 100.0, 100.0], 1.0) == 0.0


def test_metric_tracker_seeds_first_sample() -> None:
    t = MetricTracker(alpha=0.3, window=3)
    assert t.update(7.0) == (7.0, pytest.approx(1.0))
    assert t.update(7.0)[1] == pytest.approx(1.0)


def test_metric_tracker_smoothes_and_detects_volatility() -> None:
    t = MetricTracker(alpha=0.5, window=4)
    t.update(10.0)
    t.update(10.0)
    smoothed, stab = t.update(30.0)  # jump up -> low stability
    assert 10.0 < smoothed < 30.0
    assert stab < 1.0


def test_metric_tracker_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MetricTracker(alpha=0.0)
    with pytest.raises(ValueError):
        MetricTracker(window=0)


# -- heterogeneity -------------------------------------------------------


def test_normalized_difference_identical_is_zero() -> None:
    assert normalized_difference({"c": 5.0}, {"c": 5.0}, {"c": 1.0}) == 0.0


def test_normalized_difference_symmetric() -> None:
    a = {"c": 10.0, "m": 100.0}
    b = {"c": 5.0, "m": 50.0}
    w = {"c": 1.0, "m": 1.0}
    assert normalized_difference(a, b, w) == pytest.approx(normalized_difference(b, a, w))


def test_normalized_difference_zero_denominator_is_zero() -> None:
    assert normalized_difference({"c": 0.0}, {"c": 0.0}, {"c": 1.0}) == 0.0


def test_heterogeneity_single_node_is_zero() -> None:
    assert heterogeneity({"n1": {"c": 1.0}}) == {"n1": 0.0}


def test_heterogeneity_identical_nodes_are_zero() -> None:
    r = {"a": {"c": 2.0}, "b": {"c": 2.0}}
    assert heterogeneity(r) == {"a": 0.0, "b": 0.0}


def test_heterogeneity_outlier_has_higher_score() -> None:
    r = {"a": {"c": 1.0}, "b": {"c": 1.0}, "c": {"c": 9.0}}
    scores = heterogeneity(r)
    assert scores["c"] > scores["a"]
    assert 0.0 <= scores["c"] <= 1.0


# -- ActivationStats -----------------------------------------------------


def _stats(num_layers: int = 2, num_experts: int = 4) -> ActivationStats:
    return ActivationStats(num_layers=num_layers, num_experts=num_experts)


def test_record_and_marginals() -> None:
    s = _stats()
    s.record([[0, 1], [2]])
    s.record([[0, 2], [2, 3]])
    assert s.total_tokens == 2
    m = s.marginals()
    assert m[0][0] == pytest.approx(1.0)  # expert 0 selected every token at layer 0
    assert m[0][1] == pytest.approx(0.5)
    assert m[1][2] == pytest.approx(1.0)


def test_record_rejects_bad_layer_count() -> None:
    with pytest.raises(ValueError):
        _stats().record([[0]])


def test_record_rejects_out_of_range_expert() -> None:
    with pytest.raises(ValueError):
        _stats(num_experts=4).record([[0], [4]])


def test_top_experts_orders_by_frequency_then_id() -> None:
    s = _stats(num_layers=1, num_experts=4)
    s.record([[0, 3]])
    s.record([[0, 3]])
    s.record([[3, 1]])
    assert s.top_experts(0, 2) == [3, 0]  # 3 appears 3x, 0 appears 2x


def test_expected_load_is_sum_of_marginals() -> None:
    s = _stats(num_layers=1, num_experts=4)
    s.record([[0, 1]])
    s.record([[2, 3]])
    assert s.expected_load(0, [0, 2]) == pytest.approx(1.0)  # 0.5 + 0.5


def test_samples_capped() -> None:
    s = ActivationStats(num_layers=1, num_experts=4, max_samples=2)
    for i in range(5):
        s.record([[i % 4]])
    assert len(s.samples) == 2
    # counts still exact (5 tokens).
    assert s.total_tokens == 5


def test_sample_tokens_deterministic_strided() -> None:
    s = _stats(num_layers=1, num_experts=4)
    for i in range(10):
        s.record([[i % 4]])
    out = s.sample_tokens(3)
    assert out == s.sample_tokens(3)  # deterministic
    assert len(out) == 3


def test_trace_buffer_snapshot() -> None:
    buf = TraceBuffer(num_layers=2, num_experts=4)
    buf.record([[0, 1], [2]])
    snap = buf.snapshot()
    assert snap.total_tokens == 1
    assert snap.marginals()[0][0] == pytest.approx(1.0)
