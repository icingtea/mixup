"""Tests for the predictor interface, frequency baseline, and syncable state."""

from mixup_core.predictor import FrequencyPredictor, PredictorState
from mixup_core.stats import ActivationStats


def _stats() -> ActivationStats:
    s = ActivationStats(num_layers=3, num_experts=4)
    # layer 0 -> always expert 0; layer 1 -> always expert 2; layer 2 -> always expert 3.
    for _ in range(4):
        s.record([[0, 1], [2, 3], [3, 0]])
    return s


def test_frequency_predicts_next_layer_marginals() -> None:
    p = FrequencyPredictor(_stats())
    # predicting for layer 0 -> returns layer 1's marginals.
    probs = p.predict(0, None, [0])
    assert probs[2] > probs[0]  # expert 2 is the hot one at layer 1


def test_frequency_past_last_layer_is_zero() -> None:
    p = FrequencyPredictor(_stats())
    assert p.predict(2, None, [3]) == [0.0, 0.0, 0.0, 0.0]


def test_predictor_state_roundtrip() -> None:
    s = PredictorState(version=7, kind="lr", feature_dim=10, num_experts=4, weights_b64="aGk=")
    assert PredictorState.model_validate_json(s.model_dump_json()) == s


def test_predictor_state_defaults() -> None:
    s = PredictorState(version=0, kind="frequency")
    assert s.feature_dim == 0
    assert s.weights_b64 == ""
