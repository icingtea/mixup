"""Device profiles and dynamic resource state for heterogeneous placement.

These are the perception layer of the placement engine: a *static* ``DeviceProfile``
(execution type, compute/memory/bandwidth, measured expert-serving latencies) and a
*dynamic* ``ResourceState`` (utilization, free memory, in-flight load, RTT). Both are
pure pydantic — no torch, no sockets — so the cost model, placement optimizer, and
simulator can be unit-tested on the host.
"""

from enum import StrEnum

from pydantic import BaseModel

from .types import NodeId


class ExecutionType(StrEnum):
    """Whether a device runs experts on its GPU or its CPU. Fixed per device."""

    GPU = "gpu"
    CPU = "cpu"


class ProfileMode(StrEnum):
    """How placement decisions treat the (static) profile vs (dynamic) measurements.

    - ``configured``: trust the profile's declared numbers (lab emulation; deterministic).
    - ``measured``: trust online EWMA estimates (real deployment; adapts).
    - ``hybrid``: seed from the profile, correct online within bounded drift.
    """

    CONFIGURED = "configured"
    MEASURED = "measured"
    HYBRID = "hybrid"


class DeviceProfile(BaseModel):
    """Static, per-device capabilities and measured latencies.

    ``expert_fixed_ms`` / ``expert_per_ms`` are the key serving numbers: they come from
    ``mixup-bench`` (a microbenchmark of the expert forward for 1..k experts), not from a
    FLOPs estimate, so kernel-launch and memory overheads are captured. ``*_tflops`` /
    ``*_gflops`` remain as informational and for the simulator's scaling model.
    """

    node_id: NodeId
    execution: ExecutionType

    # compute capability (informational + simulator)
    cpu_cores: int = 1
    cpu_gflops: float = 10.0
    gpu_tflops: float | None = None

    # measured expert-serving latency (mixup-bench)
    expert_fixed_ms: float = 0.0
    expert_per_ms: float = 0.1

    # memory tiers
    ram_bytes: int = 0
    vram_bytes: int | None = None
    expert_budget_bytes: int | None = None
    """Bytes of expert weights this node is allowed to hold (placement memory budget).

    ``None`` = unbounded (derive from vram/ram in the planner). Set explicitly to emulate
    a memory-constrained device or reserve headroom for the dense model.
    """

    # storage / interconnect
    disk_read_mbps: float = 100.0
    disk_write_mbps: float = 100.0
    gpu_cpu_bw_mbps: float | None = None

    # leadership
    can_lead: bool = True
    lead_score: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.execution is ExecutionType.GPU


class ResourceState(BaseModel):
    """Dynamic, time-varying resource measurements for one device.

    ``rtt_ms`` is the measured round-trip time from the observing node to each peer; the
    leader maintains it from its gossip pings and uses it in the cost model.
    """

    cpu_util: float = 0.0
    gpu_util: float | None = None
    free_ram_bytes: int = 0
    free_vram_bytes: int | None = None
    active_experts: int = 0
    queue_depth: int = 0
    rtt_ms: dict[NodeId, float] = {}
