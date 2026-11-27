"""Storage tiers and per-(layer, expert) residency state.

This is the pure, torch-free core of the offloading layer (CoMoE §3.3 / §5 adapted to
CPU-only devices). A *tier* is where an expert's weights for one layer currently live.
Disk is a capacity tier only: ``can_execute`` is ``False`` for ``DISK``, and every device
reaches its execution tier through a fixed ladder.
"""

from dataclasses import dataclass

from .devices import DeviceProfile, ExecutionType


class Tier:
    """Storage tiers, ordered fastest → slowest per execution type."""

    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"

    @staticmethod
    def for_device(profile: DeviceProfile) -> list[str]:
        """The ladder for a device: GPU → CPU → disk, or CPU → disk (CPU-only)."""
        if profile.execution is ExecutionType.GPU:
            return [Tier.GPU, Tier.CPU, Tier.DISK]
        return [Tier.CPU, Tier.DISK]

    @staticmethod
    def can_execute(tier: str, profile: DeviceProfile) -> bool:
        """True if experts on ``tier`` can run directly on this device (disk cannot)."""
        if profile.execution is ExecutionType.GPU:
            return tier == Tier.GPU
        return tier == Tier.CPU


@dataclass
class ResidencyState:
    """Where one expert's weights for one layer live, plus its usage history."""

    tier: str
    bytes: int = 0
    last_used: int = 0  # monotonic access tick (recency)
    use_count: float = 0.0  # decayed activation count (frequency/importance)
    pinned: bool = False  # pinned experts are never evicted

    def touch(self, clock: int, decay: float) -> None:
        """Record an access: bump recency and decayed frequency."""
        self.last_used = clock
        self.use_count = 1.0 + decay * self.use_count
