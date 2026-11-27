"""Scenario tests: the placement policies on realistic synthetic MoE traces.

These are the simulation-side of the Phase 2 gate: a resource-aware placement must beat
the uniform baseline on a heterogeneous cluster where a fast GPU and slow CPUs share the
expert pool, and the hill-climb must never regress its frequency-greedy seed.
"""

import random

from mixup_core.placement import (
    FrequencyPolicy,
    LatencyPolicy,
    PlacementProblem,
    UniformPolicy,
    evaluate,
    violation_bytes,
)
from mixup_core.profiles import DeviceSpec
from mixup_core.tracegen import generate_trace, trace_to_stats


def _hetero_problem(num_tokens: int = 200, seed: int = 0):
    """1 fast GPU + 2 slow CPUs; 32 experts, 24 layers, top-8, Zipf activations."""
    num_layers, num_experts, top_k = 24, 32, 8
    trace = generate_trace(num_layers, num_experts, top_k, num_tokens, zipf=1.0, seed=seed)
    stats = trace_to_stats(trace, num_layers, num_experts)

    specs = {
        "gpu": DeviceSpec(
            execution="gpu", expert_fixed_ms=0.2, expert_per_ms=0.08, vram_bytes=8 << 30
        ),
        "pi1": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
        "pi2": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
    }
    profiles = {name: spec.to_profile(name) for name, spec in specs.items()}
    return PlacementProblem(
        num_experts=num_experts,
        profiles=profiles,
        stats=stats,
        rtt_ms={"gpu": 0.5, "pi1": 0.5, "pi2": 0.5},
        leader="gpu",
    )


def test_frequency_beats_uniform_on_heterogeneous_cluster() -> None:
    p = _hetero_problem()
    uniform = evaluate(p, UniformPolicy().place(p), num_samples=64, seed=1)
    freq = evaluate(p, FrequencyPolicy().place(p), num_samples=64, seed=1)
    # Hot experts land on the GPU, so the tail is much better than round-robin.
    assert freq.p95_ms < uniform.p95_ms
    assert freq.mean_ms < uniform.mean_ms


def test_latency_policy_never_worse_than_frequency() -> None:
    p = _hetero_problem()
    freq = FrequencyPolicy().place(p)
    lat = LatencyPolicy(max_iters=20, num_samples=32, seed=2).place(p)
    assert evaluate(p, lat, num_samples=64, seed=3).p95_ms <= evaluate(
        p, freq, num_samples=64, seed=3
    ).p95_ms + 1e-9


def test_policies_cover_and_are_unique() -> None:
    p = _hetero_problem()
    for policy in (UniformPolicy(), FrequencyPolicy(), LatencyPolicy(num_samples=16)):
        a = policy.place(p)
        assert set(a) == set(range(32))
        assert len(set(a.values())) <= 3  # every expert assigned, owners from the 3 nodes


def test_latency_policy_respects_budget() -> None:
    p = _hetero_problem()
    p.expert_bytes = 1_000_000
    # Total capacity 36 >= 32 experts, but each node is capped at 12.
    p.budget_bytes = {n: 12 * 1_000_000 for n in p.nodes()}
    a = LatencyPolicy(max_iters=20, num_samples=16).place(p)
    assert violation_bytes(p, a) == 0
    assert all(list(a.values()).count(n) <= 12 for n in p.nodes())


def test_deterministic_fuzz_no_regression_or_violation() -> None:
    rng = random.Random(1234)
    for _ in range(10):
        num_experts = rng.randint(8, 32)
        num_layers = rng.randint(4, 24)
        top_k = rng.randint(2, 8)
        num_tokens = rng.randint(50, 200)
        zipf = rng.uniform(0.5, 1.5)
        trace = generate_trace(
            num_layers, num_experts, top_k, num_tokens, zipf, seed=rng.randrange(1000)
        )
        stats = trace_to_stats(trace, num_layers, num_experts)
        specs = {
            "gpu": DeviceSpec(
                execution="gpu", expert_fixed_ms=0.2, expert_per_ms=0.08, vram_bytes=8 << 30
            ),
            "cpu": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
        }
        profiles = {name: spec.to_profile(name) for name, spec in specs.items()}
        p = PlacementProblem(
            num_experts=num_experts,
            profiles=profiles,
            stats=stats,
            rtt_ms={"gpu": 0.5, "cpu": 0.5},
            leader="gpu",
        )
        freq = FrequencyPolicy().place(p)
        lat = LatencyPolicy(max_iters=10, num_samples=16, seed=7).place(p)
        assert set(lat) == set(range(num_experts))
        assert set(lat.values()) <= {"gpu", "cpu"}
        # Hill-climb never regresses its seed.
        assert evaluate(p, lat, num_samples=16, seed=9).p95_ms <= evaluate(
            p, freq, num_samples=16, seed=9
        ).p95_ms + 1e-9
