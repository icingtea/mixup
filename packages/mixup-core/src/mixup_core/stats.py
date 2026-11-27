"""Activation statistics: what the placement engine needs to know about routing.

The leader observes every router's top-k selection (it runs the routers), so it can build
per-layer expert activation frequencies and a bounded sample of actual selections. The
marginals drive greedy placement; the samples let the optimizer estimate the *joint*
distribution (p95 per-node load) instead of only the mean.
"""

from collections import deque
from pathlib import Path

from pydantic import BaseModel

from .types import ExpertId

DEFAULT_MAX_SAMPLES = 1024
"""Bounded token samples kept for joint-distribution estimates; marginals stay exact."""


class ActivationStats(BaseModel):
    """Per-layer expert activation frequencies plus a bounded sample of selections."""

    num_layers: int
    num_experts: int
    tokens: int = 0
    """Total number of recorded tokens (not expert selections)."""

    counts: list[list[float]] = []
    """``counts[l][e]`` = raw number of tokens where expert ``e`` was selected at layer ``l``."""

    samples: list[list[list[ExpertId]]] = []
    """Recent ``max_samples`` tokens: ``samples[t][l]`` = selected expert ids at layer ``l``."""

    max_samples: int = DEFAULT_MAX_SAMPLES

    def model_post_init(self, __context) -> None:  # noqa: D105 - pydantic hook
        if not self.counts:
            self.counts = [[0.0] * self.num_experts for _ in range(self.num_layers)]

    def record(self, layer_experts: list[list[ExpertId]]) -> None:
        """Ingest one token's per-layer selections (``layer_experts[l]`` = selected ids)."""
        if len(layer_experts) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers, got {len(layer_experts)}"
            )
        for layer, experts in enumerate(layer_experts):
            for e in experts:
                if not 0 <= e < self.num_experts:
                    raise ValueError(f"expert {e} out of range 0..{self.num_experts - 1}")
                self.counts[layer][e] += 1.0
        self.tokens += 1
        self.samples.append(layer_experts)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]

    @property
    def total_tokens(self) -> int:
        return self.tokens

    def marginals(self) -> list[list[float]]:
        """``p[expert | layer]`` normalized by the number of recorded tokens."""
        n = self.total_tokens
        if n == 0:
            return [[0.0] * self.num_experts for _ in range(self.num_layers)]
        return [[c / n for c in row] for row in self.counts]

    def top_experts(self, layer: int, k: int) -> list[ExpertId]:
        """The ``k`` most-frequently-selected experts at ``layer``, ties broken by id."""
        order = sorted(range(self.num_experts), key=lambda e: (-self.counts[layer][e], e))
        return order[:k]

    def expected_load(self, layer: int, experts: list[ExpertId]) -> float:
        """Expected number of ``experts`` selected at ``layer`` (linearity of expectation)."""
        marginals = self.marginals()
        return sum(marginals[layer][e] for e in experts)

    def sample_tokens(self, n: int, seed: int = 0) -> list[list[list[ExpertId]]]:
        """Deterministically sample up to ``n`` tokens from the recorded samples."""
        if not self.samples or n <= 0:
            return []
        if n >= len(self.samples):
            return list(self.samples)
        # Deterministic strided subsample: spreads across the window without an RNG.
        step = len(self.samples) / n
        idx = sorted({min(len(self.samples) - 1, int(i * step)) for i in range(n)})
        return [self.samples[i] for i in idx]


class TraceBuffer:
    """Accumulates per-layer selections into an ``ActivationStats`` (used by mixup-trace)."""

    def __init__(
        self, num_layers: int, num_experts: int, max_samples: int = DEFAULT_MAX_SAMPLES
    ) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.max_samples = max_samples
        self.tokens = 0
        self.counts = [[0.0] * num_experts for _ in range(num_layers)]
        self.samples: deque[list[list[ExpertId]]] = deque(maxlen=max_samples)

    def record(self, layer_experts: list[list[ExpertId]]) -> None:
        if len(layer_experts) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layers, got {len(layer_experts)}"
            )
        for layer, experts in enumerate(layer_experts):
            for e in experts:
                self.counts[layer][e] += 1.0
        self.tokens += 1
        self.samples.append(layer_experts)

    def snapshot(self) -> ActivationStats:
        return ActivationStats(
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            tokens=self.tokens,
            counts=self.counts,
            samples=list(self.samples),
            max_samples=self.max_samples,
        )


def load_trace_file(
    path: str | Path, max_samples: int = DEFAULT_MAX_SAMPLES
) -> ActivationStats:
    """Load a ``mixup-trace`` JSONL file (one ``{"token", "layers"}`` per line) into stats.

    ``num_layers`` / ``num_experts`` are inferred from the first token; ``max_samples`` caps
    the retained token samples (the marginals stay exact for all tokens).
    """
    import json

    tokens: list[list[list[ExpertId]]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        tokens.append(json.loads(line)["layers"])

    if not tokens:
        raise ValueError(f"trace file {path} is empty")

    num_layers = len(tokens[0])
    num_experts = max(max(layer) for token in tokens for layer in token) + 1
    stats = ActivationStats(
        num_layers=num_layers, num_experts=num_experts, max_samples=max_samples
    )
    for token in tokens:
        stats.record(token)
    return stats
