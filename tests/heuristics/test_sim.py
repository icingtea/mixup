"""Tests for the deterministic simulator."""

import pytest

from mixup_core.devices import DeviceProfile
from mixup_core.placement import FrequencyPolicy, evaluate
from mixup_core.sim import Simulator
from mixup_core.stats import ActivationStats


def _gpu(node: str, per: float = 0.5) -> DeviceProfile:
    return DeviceProfile(node_id=node, execution="gpu", expert_fixed_ms=1.0, expert_per_ms=per)


def _cpu(node: str, per: float = 3.0) -> DeviceProfile:
    return DeviceProfile(node_id=node, execution="cpu", expert_fixed_ms=2.0, expert_per_ms=per)


def _problem(num_experts: int = 4, num_layers: int = 3) -> tuple:
    from mixup_core.placement import PlacementProblem

    stats = ActivationStats(num_layers=num_layers, num_experts=num_experts)
    for _ in range(6):
        stats.record([[0, 1], [2, 3], [0, 2]])
    profiles = {"a": _gpu("a"), "b": _cpu("b")}
    p = PlacementProblem(
        num_experts=num_experts,
        profiles=profiles,
        stats=stats,
        rtt_ms={"a": 0.5, "b": 0.5},
        leader="a",
        dense_ms=2.0,
    )
    return p, stats


def test_single_stream_matches_evaluate() -> None:
    p, _ = _problem()
    assignment = FrequencyPolicy().place(p)
    # queueing=False → simulator equals the analytic evaluate() exactly.
    sim = Simulator(p).run(assignment, num_samples=16, seed=1, queueing=False)
    analytic = evaluate(p, assignment, num_samples=16, seed=1)
    assert sim.stats.p95_ms == pytest.approx(analytic.p95_ms)
    assert sim.stats.mean_ms == pytest.approx(analytic.mean_ms)


def test_simulator_is_deterministic() -> None:
    p, _ = _problem()
    assignment = FrequencyPolicy().place(p)
    sim = Simulator(p)
    assert sim.run(assignment, seed=7) == sim.run(assignment, seed=7)


def test_single_stream_has_no_queueing() -> None:
    p, _ = _problem()
    assignment = FrequencyPolicy().place(p)
    assert Simulator(p).run(assignment, num_samples=8, seed=2).queue_ms == 0.0


def test_per_node_served_reflects_assignment() -> None:
    p, stats = _problem()
    # Force all experts onto node "a".
    assignment = {0: "a", 1: "a", 2: "a", 3: "a"}
    result = Simulator(p).run(assignment, num_samples=16, seed=3)
    assert result.per_node_served["a"] > 0
    assert result.per_node_served.get("b", 0) == 0


def test_slow_node_dominates_layer_time() -> None:
    p, _ = _problem()
    # Node "b" (CPU) serving everything is slower than "a" (GPU).
    result = Simulator(p).run({0: "b", 1: "b", 2: "b", 3: "b"}, num_samples=16, seed=4)
    fast = Simulator(p).run({0: "a", 1: "a", 2: "a", 3: "a"}, num_samples=16, seed=4)
    assert result.stats.mean_ms > fast.stats.mean_ms
