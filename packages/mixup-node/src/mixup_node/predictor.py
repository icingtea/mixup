"""Trained next-layer expert predictors: logistic regression and a 2-layer MLP.

Phase 6 (CoMoE §2.2.5, adapted to CPU/GPU). Each predicts, for the *next* layer, the
probability that each expert is in the top-k, from the current hidden state and current
experts: ``p(e^{l+1}) = f([h_t, one_hot(e_l), 1])``. Trained online with binary
cross-entropy SGD against the router's actual next-layer top-k. Weights serialize to a
``PredictorState`` (base64) so the leader can sync them every ``kv_sync_every`` tokens.

Torch. Runs only on the leader (the only place with ``h_t`` and the router).
"""

import base64
import io

import torch
from torch import Tensor

from mixup_core.predictor import PredictorState


def features(
    hidden_state: Tensor, current_experts: list[int], num_experts: int
) -> Tensor:
    """Feature vector ``[h_t, one_hot(e_l), 1]`` for the predictor input."""
    h = hidden_state.reshape(-1).float()
    one_hot = torch.zeros(num_experts, dtype=torch.float32, device=h.device)
    for e in current_experts:
        one_hot[e] = 1.0
    return torch.cat([h, one_hot, torch.tensor([1.0], device=h.device)])


def _labels(next_experts: list[int], num_experts: int) -> Tensor:
    y = torch.zeros(num_experts)
    for e in next_experts:
        y[e] = 1.0
    return y


class LRPredictor:
    """32 independent binary logistic regressions over ``[h_t, one_hot(e_l), 1]``."""

    kind = "lr"

    def __init__(self, feature_dim: int, num_experts: int, lr: float = 0.05) -> None:
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.lr = lr
        self.W = torch.zeros(num_experts, feature_dim)  # [num_experts, feature_dim]
        self.samples = 0
        self.top1_hits = 0
        self.top3_hits = 0

    def record_accuracy(self, probs: Tensor, actual: list[int]) -> None:
        """Track top-1/top-3 accuracy against the actual next-layer top-k."""
        self.samples += 1
        ranked = sorted(range(probs.shape[0]), key=lambda e: -probs[e].item())
        if ranked[0] in actual:
            self.top1_hits += 1
        if any(e in actual for e in ranked[:3]):
            self.top3_hits += 1

    def _sync_device(self, device) -> None:
        if self.W.device != device:
            self.W = self.W.to(device)

    def predict(self, x: Tensor) -> Tensor:
        """``x`` is a feature vector; returns per-expert probabilities."""
        self._sync_device(x.device)
        return torch.sigmoid(self.W @ x)

    def update(self, x: Tensor, y: Tensor) -> float:
        """One SGD step; returns the loss (for logging)."""
        self._sync_device(x.device)
        y = y.to(x.device)
        p = self.predict(x)
        loss = torch.nn.functional.binary_cross_entropy(p, y)
        grad = (p - y).unsqueeze(1) @ x.unsqueeze(0)  # [num_experts, feature_dim]
        self.W = self.W - self.lr * grad
        return loss.item()

    def to_state(self, version: int) -> PredictorState:
        buf = io.BytesIO()
        torch.save(self.W.cpu(), buf)
        return PredictorState(
            version=version,
            kind=self.kind,
            feature_dim=self.feature_dim,
            num_experts=self.num_experts,
            weights_b64=base64.b64encode(buf.getvalue()).decode(),
        )

    @classmethod
    def from_state(cls, state: PredictorState) -> "LRPredictor":
        model = cls(state.feature_dim, state.num_experts)
        model.W = torch.load(
            io.BytesIO(base64.b64decode(state.weights_b64)), map_location="cpu"
        )
        return model


class MLPPredictor:
    """CoMoE's 2-layer MLP: ``sigmoid(W2 @ relu(W1 @ [h_t, one_hot(e_l), 1] + b1) + b2)``."""

    kind = "mlp"

    def __init__(
        self, feature_dim: int, num_experts: int, hidden_dim: int = 64, lr: float = 0.01
    ) -> None:
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.W1 = torch.randn(hidden_dim, feature_dim) * 0.1
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(num_experts, hidden_dim) * 0.1
        self.b2 = torch.zeros(num_experts)
        self.samples = 0
        self.top1_hits = 0
        self.top3_hits = 0

    def record_accuracy(self, probs: Tensor, actual: list[int]) -> None:
        self.samples += 1
        ranked = sorted(range(probs.shape[0]), key=lambda e: -probs[e].item())
        if ranked[0] in actual:
            self.top1_hits += 1
        if any(e in actual for e in ranked[:3]):
            self.top3_hits += 1

    def _sync_device(self, device) -> None:
        if self.W1.device != device:
            self.W1 = self.W1.to(device)
            self.b1 = self.b1.to(device)
            self.W2 = self.W2.to(device)
            self.b2 = self.b2.to(device)

    def predict(self, x: Tensor) -> Tensor:
        self._sync_device(x.device)
        h = torch.relu(self.W1 @ x + self.b1)
        return torch.sigmoid(self.W2 @ h + self.b2)

    def update(self, x: Tensor, y: Tensor) -> float:
        self._sync_device(x.device)
        y = y.to(x.device)
        h = torch.relu(self.W1 @ x + self.b1)
        p = torch.sigmoid(self.W2 @ h + self.b2)
        loss = torch.nn.functional.binary_cross_entropy(p, y)
        # Manual backprop (2-layer, 32 outputs) — simple enough to hand-roll.
        d = (p - y)  # [num_experts]
        dW2 = d.unsqueeze(1) @ h.unsqueeze(0)
        db2 = d
        dh = self.W2.t() @ d  # [hidden_dim]
        dh = dh * (h > 0).float()
        dW1 = dh.unsqueeze(1) @ x.unsqueeze(0)
        db1 = dh
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return loss.item()

    def to_state(self, version: int) -> PredictorState:
        buf = io.BytesIO()
        torch.save(
            {"W1": self.W1.cpu(), "b1": self.b1.cpu(), "W2": self.W2.cpu(), "b2": self.b2.cpu()},
            buf,
        )
        return PredictorState(
            version=version,
            kind=self.kind,
            feature_dim=self.feature_dim,
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            weights_b64=base64.b64encode(buf.getvalue()).decode(),
        )

    @classmethod
    def from_state(cls, state: PredictorState) -> "MLPPredictor":
        model = cls(state.feature_dim, state.num_experts, state.hidden_dim or 64)
        d = torch.load(
            io.BytesIO(base64.b64decode(state.weights_b64)), map_location="cpu"
        )
        model.W1, model.b1, model.W2, model.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return model
