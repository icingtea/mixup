"""The predictor interface, the frequency baseline, and the syncable state.

Phase 6. The *trained* predictors (logistic regression, 2-layer MLP) live in
``mixup_node.predictor`` (they need torch); this core module holds what is torch-free:

- ``Predictor`` — the interface (predict next-layer expert probabilities).
- ``FrequencyPredictor`` — the trivial baseline (per-layer marginals).
- ``PredictorState`` — the versioned, serializable weights the leader syncs to followers
  every ``kv_sync_every`` tokens so a promoted leader resumes prefetching.
"""

from typing import Protocol

from pydantic import BaseModel

from .stats import ActivationStats
from .types import ExpertId


class Predictor(Protocol):
    """Predicts the probability that each expert is selected at the *next* layer."""

    def predict(
        self, layer: int, hidden_state, current_experts: list[ExpertId]
    ) -> list[float]: ...


class FrequencyPredictor:
    """Baseline: the per-layer marginal activation frequency (no learned model)."""

    def __init__(self, stats: ActivationStats) -> None:
        self.stats = stats
        self._marginals = stats.marginals()

    def predict(
        self, layer: int, hidden_state, current_experts: list[ExpertId]
    ) -> list[float]:
        nxt = layer + 1
        if nxt >= self.stats.num_layers:
            return [0.0] * self.stats.num_experts
        return list(self._marginals[nxt])


class PredictorState(BaseModel):
    """A versioned snapshot of the leader's predictor weights, for failover sync.

    ``weights_b64`` is the model-dependent serialized weights (empty for the frequency
    baseline, which is derived from the activation stats instead).
    """

    version: int
    kind: str
    """``"frequency"``, ``"lr"``, or ``"mlp"``."""

    feature_dim: int = 0
    num_experts: int = 0
    hidden_dim: int = 0
    weights_b64: str = ""
