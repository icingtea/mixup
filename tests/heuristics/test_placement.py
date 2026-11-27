"""Tests for placement policies and the latency objective."""

import pytest

from mixup_core.cost import CostModel
from mixup_core.devices import DeviceProfile
from mixup_core.placement import (
    FrequencyPolicy,
    LatencyPolicy,
    LatencyStats,
    PlacementProblem,
    UniformPolicy,
    activation_mass,
    churn_ms,
    evaluate,
    moved_experts,
    objective,
    violation_bytes,
)
from mixup_core.stats import ActivationStats


def _gpu(node: str, per: float = 0.1) -> DeviceProfile:
    return DeviceProfile(node_id=node, execution="gpu", expert_fixed_ms=0.5, expert_per_ms=per)


def _cpu(node: str, per: float = 1.0) -> DeviceProfile:
    return DeviceProfile(node_id=node, execution="cpu", expert_fixed_ms=1.0, expert_per_ms=per)


def _stats(num_layers: int = 2, num_experts: int = 4) -> ActivationStats:
    return ActivationStats(num_layers=num_layers, num_experts=num_experts)


def _problem(
    num_experts: int = 4,
    num_layers: int = 2,
    hot: int = 0,
    budget: dict[str, int] | None = None,
    prev: dict[int, str] | None = None,
) -> PlacementProblem:
    stats = _stats(num_layers, num_experts)
    for _ in range(8):
        stats.record([[hot, 1], [hot, 2]])  # expert `hot` selected at every layer
    profiles = {"fast": _gpu("fast"), "slow": _cpu("slow")}
    return PlacementProblem(
        num_experts=num_experts,
        profiles=profiles,
        stats=stats,
        rtt_ms={"fast": 0.0, "slow": 0.0},
        leader="fast",
        expert_bytes=1_000_000,
        budget_bytes=budget,
        prev=prev,
    )


def test_uniform_round_robin() -> None:
    p = _problem()
    assert UniformPolicy().place(p) == {0: "fast", 1: "slow", 2: "fast", 3: "slow"}


def test_frequency_puts_hot_expert_on_fast_node() -> None:
    p = _problem(hot=3)
    a = FrequencyPolicy().place(p)
    assert a[3] == "fast"


def test_frequency_is_deterministic() -> None:
    p = _problem()
    assert FrequencyPolicy().place(p) == FrequencyPolicy().place(p)


def test_frequency_respects_capacity() -> None:
    # fast can hold only 1 expert (1 MB budget, 1 MB expert).
    p = _problem(budget={"fast": 1_000_000, "slow": 10_000_000})
    a = FrequencyPolicy().place(p)
    owners = [a[e] for e in range(4)]
    assert owners.count("fast") == 1


def test_uniform_respects_capacity() -> None:
    p = _problem(budget={"fast": 1_000_000, "slow": 10_000_000})
    a = UniformPolicy().place(p)
    assert list(a.values()).count("fast") == 1


def test_latency_policy_not_worse_than_frequency() -> None:
    p = _problem()
    freq = FrequencyPolicy().place(p)
    lat = LatencyPolicy(max_iters=10, num_samples=16).place(p)
    assert objective(p, lat, 32) <= objective(p, freq, 32) + 1e-9


def test_latency_policy_deterministic() -> None:
    p = _problem()
    policy = LatencyPolicy(max_iters=10, num_samples=16, seed=3)
    assert policy.place(p) == policy.place(p)


def test_evaluate_returns_stats_shape() -> None:
    p = _problem()
    stats = evaluate(p, FrequencyPolicy().place(p), num_samples=4)
    assert isinstance(stats, LatencyStats)
    assert stats.num_samples == 4
    assert stats.p50_ms <= stats.p95_ms <= stats.max_ms


def test_slow_node_makes_tail_worse() -> None:
    # Put the hot expert on the slow CPU node: p95 should exceed the fast-node placement.
    p = _problem(hot=3)
    slow_assignment = {0: "slow", 1: "fast", 2: "slow", 3: "slow"}
    fast_assignment = {0: "slow", 1: "fast", 2: "slow", 3: "fast"}
    slow_eval = evaluate(p, slow_assignment, num_samples=16)
    fast_eval = evaluate(p, fast_assignment, num_samples=16)
    assert slow_eval.p95_ms >= fast_eval.p95_ms


def test_activation_mass_sums_layers() -> None:
    p = _problem(hot=0)
    masses = activation_mass(p.stats)
    assert masses[0] == pytest.approx(2.0)  # selected in both layers, every token
    assert masses[1] == pytest.approx(1.0)  # layer 0 only


def test_churn_counts_moved_experts() -> None:
    p = _problem(prev={0: "fast", 1: "fast", 2: "slow", 3: "slow"})
    a = {0: "slow", 1: "fast", 2: "slow", 3: "slow"}
    assert moved_experts(p, a) == [0]
    assert churn_ms(p, a) > 0.0


def test_churn_zero_when_no_prev() -> None:
    p = _problem(prev=None)
    assert churn_ms(p, {0: "fast", 1: "slow", 2: "fast", 3: "slow"}) == 0.0


def test_violation_bytes_sums_over_budget() -> None:
    p = _problem(budget={"fast": 1_000_000, "slow": 1_000_000})
    a = {0: "fast", 1: "fast", 2: "slow", 3: "slow"}
    # fast holds 2 experts (2 MB over 1 MB) and slow likewise -> 2 MB total violation.
    assert violation_bytes(p, a) == 2_000_000


def test_violation_zero_within_budget() -> None:
    p = _problem(budget={"fast": 10_000_000, "slow": 10_000_000})
    a = {0: "fast", 1: "fast", 2: "slow", 3: "slow"}
    assert violation_bytes(p, a) == 0


def test_cost_model_is_built() -> None:
    p = _problem()
    assert isinstance(p.cost, CostModel)
