"""Resource smoothing, stability, and heterogeneity — CoMoE §2.2.2, adapted.

Pure math, no torch. The placement engine consumes the *smoothed* resource values and a
per-resource *stability* score, and a per-device *heterogeneity* score for conservative
tie-breaking. Numerical details the report leaves open (window length, clipping, zero
denominator) are resolved here and unit-tested.
"""

import math
from collections import deque
from collections.abc import Iterable


def ewma(prev: float, value: float, alpha: float) -> float:
    """One exponentially weighted moving average step: ``alpha*value + (1-alpha)*prev``.

    ``alpha`` is clamped to ``(0, 1]``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return alpha * value + (1.0 - alpha) * prev


def volatility(values: Iterable[float], smoothed: float) -> float:
    """Population std-dev of ``values`` around the smoothed level ``smoothed``."""
    vs = list(values)
    if not vs:
        return 0.0
    var = sum((v - smoothed) ** 2 for v in vs) / len(vs)
    return math.sqrt(var)


def stability(values: Iterable[float], smoothed: float) -> float:
    """CoMoE stability ``S_r = 1 - sigma/smoothed``, clipped to ``[0, 1]``.

    A value near 1 means stable; near 0 means sharp fluctuation. The report does not
    specify zero-denominator handling: we treat a smoothed level at or below zero as
    maximally unstable (0), and a perfect (zero-volatility) series as fully stable (1).
    """
    if smoothed <= 0.0:
        return 0.0
    s = 1.0 - volatility(values, smoothed) / smoothed
    return max(0.0, min(1.0, s))


class MetricTracker:
    """Tracks one scalar resource indicator: EWMA-smoothed value + rolling stability.

    ``update(value)`` returns ``(smoothed, stability)`` after ingesting the sample. The
    rolling window holds the most recent ``window`` raw samples used for volatility.
    """

    def __init__(self, alpha: float = 0.3, window: int = 20) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.alpha = alpha
        self.window = window
        self.smoothed: float | None = None
        self._recent: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> tuple[float, float]:
        """Ingest ``value``; return ``(smoothed, stability)``.

        The first sample seeds the smoothed value (no smoothing history yet); stability is
        computed over the samples seen so far (fewer than ``window`` early on).
        """
        self._recent.append(value)
        self.smoothed = value if self.smoothed is None else ewma(self.smoothed, value, self.alpha)
        return self.smoothed, stability(self._recent, self.smoothed)

    @property
    def recent(self) -> list[float]:
        return list(self._recent)


def normalized_difference(
    a: dict[str, float], b: dict[str, float], weights: dict[str, float]
) -> float:
    """Weighted, normalized resource difference ``D_ij`` between two devices.

    ``weights`` map resource-name -> weight; weights are normalized to sum to 1. Each
    dimension contributes ``|a-b| / max(a, b)``; a zero ``max`` contributes 0 (both have
    none of that resource).
    """
    names = sorted(weights)
    total = sum(weights[n] for n in names)
    if total <= 0.0:
        return 0.0
    diff = 0.0
    for name in names:
        x, y = a.get(name, 0.0), b.get(name, 0.0)
        denom = max(abs(x), abs(y))
        if denom == 0.0:
            continue
        diff += weights[name] * abs(x - y) / denom
    return diff / total


def heterogeneity(
    resources: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-device heterogeneity ``H_d = mean over peers of D_dj``.

    ``resources`` maps node-id -> resource-name -> value. ``weights`` default to uniform.
    Returns node-id -> score in ``[0, 1]``; a device with no peers gets 0.
    """
    nodes = sorted(resources)
    if len(nodes) <= 1:
        return {n: 0.0 for n in nodes}
    names = sorted({name for vals in resources.values() for name in vals})
    w = weights or {name: 1.0 / len(names) for name in names}
    scores: dict[str, float] = {}
    for node in nodes:
        total = 0.0
        for peer in nodes:
            if peer == node:
                continue
            total += normalized_difference(resources[node], resources[peer], w)
        scores[node] = total / (len(nodes) - 1)
    return scores
