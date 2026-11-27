"""Residency decisions: retention, eviction, and prefetch scoring + a byte-budgeted manager.

This is the pure core of the multi-tier offloading loop (CoMoE §3.6 / §5, adapted to
CPU-only devices). The node layer applies these decisions to actual tensors; this module
only computes *what to keep, evict, and prefetch* under a per-tier byte budget.

The report leaves several formula directions/normalizations open; we resolve them to be
monotonic and defensible (see the module docstrings and ``docs/plan/phase-10-...``):

- retention: higher = keep resident (more frequent *and* more expensive to reload).
- eviction: higher = evict first (less predicted, less recent, less important).
- prefetch: lower threshold = more aggressive prefetch (more free memory *or* higher
  stability lowers it — matching the report's prose, not its sign-flipped formula).
"""

from dataclasses import dataclass, field

from .cost import load_ms
from .devices import DeviceProfile
from .tiers import ResidencyState, Tier

# -- score functions ------------------------------------------------------


def retention_score(
    freq: float,
    load_ms_: float,
    gamma: float,
    load_scale: float = 1.0,
) -> float:
    """``P_offload``: higher = keep resident.

    Trades activation frequency against retrieval cost; ``load_scale`` normalizes the load
    time (use the max load across candidates) so the two terms are comparable.
    """
    if load_scale <= 0:
        load_scale = 1.0
    return gamma * freq + (1.0 - gamma) * min(1.0, load_ms_ / load_scale)


def eviction_score(
    predicted_prob: float,
    recency: float,
    importance: float,
    delta: float,
    lambd: float,
) -> float:
    """``S_evict``: higher = evict first.

    Combines predicted inactivity (``1 - p``) with inverse recency and inverse importance.
    ``recency``/``importance`` are clamped away from zero to keep the inverses finite.
    """
    r = max(recency, 1e-6)
    i = max(importance, 1e-6)
    return (1.0 - delta) * (1.0 - predicted_prob) + delta * (lambd / r + (1.0 - lambd) / i)


def prefetch_threshold(
    theta_base: float,
    stability: float,
    free_ratio: float,
    theta_mem: float = 1.0,
) -> float:
    """``θ_pre``: prefetch an expert when its predicted probability clears this.

    Text-consistent form: more free memory (``free_ratio`` ↑) or higher stability lowers
    the threshold (more prefetch); instability or a full tier raises it (more conservative).
    ``stability`` is clamped to a small positive floor to avoid a zero denominator.
    """
    s = max(0.05, min(1.0, stability))
    f = max(0.0, free_ratio)
    return theta_base / (s * (1.0 + theta_mem * f))


# -- the byte-budgeted residency manager ----------------------------------


@dataclass
class ResidencyConfig:
    """Tunables for the residency scores (the report's γ/δ/λ plus defaults)."""

    gamma: float = 0.6
    delta: float = 0.5
    lambd: float = 0.5
    theta_base: float = 0.05
    theta_mem: float = 1.0
    decay: float = 0.9
    """EWMA decay for the frequency/recency estimate."""


@dataclass
class Residency:
    """Tracks per-(layer, expert) tiers under per-tier byte budgets, and orders
    evictions/prefetches. Pure and deterministic."""

    profile: DeviceProfile
    expert_bytes_per_layer: int
    budget_bytes: dict[str, int] = field(default_factory=dict)
    config: ResidencyConfig = field(default_factory=ResidencyConfig)
    states: dict[tuple[int, int], ResidencyState] = field(default_factory=dict)
    clock: int = 0

    @property
    def tiers(self) -> list[str]:
        return Tier.for_device(self.profile)

    # -- accounting --------------------------------------------------------

    def used_bytes(self, tier: str) -> int:
        return sum(s.bytes for s in self.states.values() if s.tier == tier)

    def free_bytes(self, tier: str) -> int:
        budget = self.budget_bytes.get(tier)
        if budget is None:
            return 1 << 60
        return max(0, budget - self.used_bytes(tier))

    def free_ratio(self, tier: str) -> float:
        budget = self.budget_bytes.get(tier)
        if not budget:
            return 1.0
        return self.free_bytes(tier) / budget

    # -- state transitions -------------------------------------------------

    def place(self, layer: int, expert: int, tier: str, bytes_: int) -> None:
        key = (layer, expert)
        if key in self.states:
            self.states[key].tier = tier
            self.states[key].bytes = bytes_
        else:
            self.states[key] = ResidencyState(tier=tier, bytes=bytes_)

    def touch(self, layer: int, expert: int) -> None:
        self.clock += 1
        self.states[(layer, expert)].touch(self.clock, self.config.decay)

    # -- decisions ---------------------------------------------------------

    def recency(self, key: tuple[int, int]) -> float:
        state = self.states[key]
        return 1.0 / (self.clock - state.last_used + 1)

    def eviction_order(self, tier: str, predicted_prob: dict) -> list[tuple[int, int]]:
        """Keys in ``tier``, sorted by eviction score descending (evict first), excluding
        pinned experts."""
        cfg = self.config
        keys = [k for k, s in self.states.items() if s.tier == tier and not s.pinned]

        def score(key: tuple[int, int]) -> float:
            s = self.states[key]
            p = predicted_prob.get(key, 0.0)
            importance = s.use_count / (s.use_count + 1.0) if s.use_count else 0.0
            return eviction_score(p, self.recency(key), importance, cfg.delta, cfg.lambd)

        return sorted(keys, key=score, reverse=True)

    def retention_order(self, tier: str, freq: dict, load_scale: float) -> list[tuple[int, int]]:
        """Keys in ``tier`` sorted by retention score descending (keep first)."""
        cfg = self.config

        def score(key: tuple[int, int]) -> float:
            s = self.states[key]
            return retention_score(
                freq.get(key, 0.0), load_ms(self.profile, s.tier, s.bytes), cfg.gamma, load_scale
            )

        return sorted(
            (k for k, s in self.states.items() if s.tier == tier and not s.pinned),
            key=score,
            reverse=True,
        )

    def ensure_room(
        self, tier: str, need_bytes: int, predicted_prob: dict
    ) -> list[tuple[int, int]]:
        """Evict the lowest-retention experts in ``tier`` until ``need_bytes`` fit.

        Returns the keys to evict (the node layer moves them to the next-slower tier).
        """
        deficit = need_bytes - self.free_bytes(tier)
        if deficit <= 0:
            return []
        order = self.eviction_order(tier, predicted_prob)
        to_evict: list[tuple[int, int]] = []
        freed = 0
        for key in order:
            to_evict.append(key)
            freed += self.states[key].bytes
            if freed >= deficit:
                break
        return to_evict
