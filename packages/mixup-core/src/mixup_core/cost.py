"""Latency cost model for expert placement and the simulator.

Converts per-device profiles + a routing/network state into per-layer and per-token
latency estimates. Compute is ``fixed + n_experts * per_expert`` (from ``mixup-bench``),
network is RTT, and tier-loading cost follows the report's storage ladders (§3.5):

- GPU: resident (GPU) → 0; CPU → ``bytes / PCIe``; disk → ``bytes / disk + bytes / PCIe``.
- CPU: resident (CPU) → 0; disk → ``bytes / disk``.

Disk is never an execution tier; a load always ends on the execution tier (GPU or CPU).
"""

from .devices import DeviceProfile, ExecutionType

_BYTES_PER_MB = 1_000_000.0
_MS_PER_S = 1_000.0


def _transfer_ms(bytes_: float, mbps: float) -> float:
    """Time to move ``bytes_`` over a link of ``mbps`` MB/s, in milliseconds."""
    if mbps is None or mbps <= 0:
        return 0.0
    return bytes_ / (mbps * _BYTES_PER_MB) * _MS_PER_S


class CostModel:
    """Latency model over a set of device profiles."""

    def __init__(self, profiles: dict[str, DeviceProfile]) -> None:
        self.profiles = profiles

    def compute_ms(self, node: str, n_experts: int) -> float:
        """Expert computation for one request serving ``n_experts`` experts (batch 1)."""
        p = self.profiles[node]
        return p.expert_fixed_ms + n_experts * p.expert_per_ms

    def network_ms(self, rtt_ms: float) -> float:
        """Round-trip dispatch cost (serialization of a 1-token payload is negligible)."""
        return rtt_ms

    def load_ms(self, node: str, from_tier: str, bytes_: float) -> float:
        """Time to bring one expert (``bytes_``) from ``from_tier`` to the execution tier.

        ``from_tier`` is ``"gpu"``/``"cpu"``/``"disk"``; ``"gpu"`` is only valid on a GPU
        device. Raises on an unknown tier or a GPU device missing its PCIe bandwidth.
        """
        p = self.profiles[node]
        if from_tier not in ("gpu", "cpu", "disk"):
            raise ValueError(f"unknown tier {from_tier!r}")

        if p.execution is ExecutionType.GPU:
            if from_tier == "gpu":
                return 0.0
            if p.gpu_cpu_bw_mbps is None or p.gpu_cpu_bw_mbps <= 0:
                raise ValueError(f"GPU node {node!r} has no gpu_cpu_bw_mbps")
            cpu_gpu = _transfer_ms(bytes_, p.gpu_cpu_bw_mbps)
            if from_tier == "cpu":
                return cpu_gpu
            return _transfer_ms(bytes_, p.disk_read_mbps) + cpu_gpu

        # CPU-only: CPU memory is the execution tier.
        if from_tier == "cpu":
            return 0.0
        if from_tier == "gpu":
            raise ValueError(f"CPU node {node!r} has no GPU tier")
        return _transfer_ms(bytes_, p.disk_read_mbps)

    def layer_ms(self, loads: dict[str, int], rtt_ms: dict[str, float]) -> float:
        """One layer's dispatch+compute: the max over the nodes serving it.

        ``loads`` maps node-id -> number of experts it serves this layer; a node serves a
        request only when ``loads[node] > 0``. Returns 0 when nothing is dispatched.
        """
        worst = 0.0
        for node, n in loads.items():
            if n <= 0:
                continue
            t = self.network_ms(rtt_ms.get(node, 0.0)) + self.compute_ms(node, n)
            worst = max(worst, t)
        return worst

    def token_ms(
        self,
        layers: list[dict[str, int]],
        rtt_ms: dict[str, float],
        dense_ms: float = 0.0,
    ) -> float:
        """One token: dense leader work plus the sum of per-layer expert dispatch."""
        return dense_ms + sum(self.layer_ms(loads, rtt_ms) for loads in layers)
