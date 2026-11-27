"""Small numeric helpers shared across the runtime (pure, torch-free)."""


def fit_linear(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of ``y = intercept + slope * x`` over ``(x, y)`` samples.

    Returns ``(intercept, slope)``, each clamped to non-negative. Used to fit the expert
    serving model ``t(n) = fixed + per * n`` from measured ``(n_experts, ms)`` points.
    """
    n = len(samples)
    if n < 2:
        return 0.0, 0.0
    sx = sum(x for x, _ in samples)
    sy = sum(y for _, y in samples)
    sxx = sum(x * x for x, _ in samples)
    sxy = sum(x * y for x, y in samples)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 0.0
    intercept = (sy - slope * sx) / n if n else 0.0
    return max(0.0, intercept), max(0.0, slope)
