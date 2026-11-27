"""Synthetic MoE activation-trace generation for placement validation.

Real MoE routing concentrates on a few "hot" experts per layer with a Zipf-like
popularity, but the specific ranking varies per layer. ``generate_trace`` reproduces that
shape deterministically so the placement engine can be tested and benchmarked on
realistic activation patterns without hardware. It feeds ``ActivationStats`` and the
simulator.
"""

import random

from .types import ExpertId


def _weighted_sample_without_replacement(
    order: list[int],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[ExpertId]:
    """Sample ``k`` distinct experts from ``order`` by ``weights``, without replacement."""
    remaining = list(zip(order, weights, strict=False))
    chosen: list[ExpertId] = []
    for _ in range(min(k, len(remaining))):
        total = sum(w for _, w in remaining)
        r = rng.random() * total
        acc = 0.0
        for i, (e, w) in enumerate(remaining):
            acc += w
            if r <= acc:
                chosen.append(e)
                remaining.pop(i)
                break
    return chosen


def generate_trace(
    num_layers: int,
    num_experts: int,
    top_k: int,
    num_tokens: int,
    zipf: float = 1.0,
    seed: int = 0,
) -> list[list[list[ExpertId]]]:
    """Generate ``num_tokens`` tokens, each ``num_layers`` layers of ``top_k`` expert ids.

    Each layer gets its own random popularity ranking with Zipf exponent ``zipf``; a higher
    exponent concentrates routing on fewer experts. Fully deterministic given ``seed``.
    """
    rng = random.Random(seed)
    layers: list[tuple[list[int], list[float]]] = []
    for _ in range(num_layers):
        order = list(range(num_experts))
        rng.shuffle(order)
        weights = [1.0 / ((i + 1) ** zipf) for i in range(num_experts)]
        layers.append((order, weights))

    trace: list[list[list[ExpertId]]] = []
    for _ in range(num_tokens):
        token = [
            _weighted_sample_without_replacement(order, weights, top_k, rng)
            for order, weights in layers
        ]
        trace.append(token)
    return trace


def trace_to_stats(
    trace: list[list[list[ExpertId]]],
    num_layers: int,
    num_experts: int,
    max_samples: int = 1024,
):
    """Fold a trace into an ``ActivationStats``."""
    from .stats import ActivationStats

    stats = ActivationStats(
        num_layers=num_layers, num_experts=num_experts, max_samples=max_samples
    )
    for token in trace:
        stats.record(token)
    return stats
