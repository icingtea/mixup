"""Concrete device profiles and a compact spec for building them.

``DeviceSpec`` is the human-writable form (GFLOPS, memory, measured latencies); it maps to
a full ``DeviceProfile`` via ``to_profile``. ``TESTBED`` holds placeholder specs for the
five-device lab (5060 + 3070 + 3060 + 2 Pi); the ``expert_*_ms`` numbers are placeholders
to be replaced by ``mixup-bench`` calibration on the real devices.
"""

from pydantic import BaseModel

from .devices import DeviceProfile, ExecutionType

GB = 1 << 30
MB = 1 << 20


class DeviceSpec(BaseModel):
    """Compact, human-writable device description."""

    execution: ExecutionType
    cpu_cores: int = 4
    cpu_gflops: float = 20.0
    gpu_tflops: float | None = None
    expert_fixed_ms: float = 0.0
    expert_per_ms: float = 1.0
    ram_bytes: int = 8 * GB
    vram_bytes: int | None = None
    disk_read_mbps: float = 100.0
    disk_write_mbps: float = 100.0
    gpu_cpu_bw_mbps: float | None = None
    can_lead: bool = True
    lead_score: int = 0

    def to_profile(self, node_id: str) -> DeviceProfile:
        return DeviceProfile(
            node_id=node_id,
            execution=self.execution,
            cpu_cores=self.cpu_cores,
            cpu_gflops=self.cpu_gflops,
            gpu_tflops=self.gpu_tflops,
            expert_fixed_ms=self.expert_fixed_ms,
            expert_per_ms=self.expert_per_ms,
            ram_bytes=self.ram_bytes,
            vram_bytes=self.vram_bytes,
            disk_read_mbps=self.disk_read_mbps,
            disk_write_mbps=self.disk_write_mbps,
            gpu_cpu_bw_mbps=self.gpu_cpu_bw_mbps,
            can_lead=self.can_lead,
            lead_score=self.lead_score,
        )


# Placeholder specs for the target testbed. The expert_*_ms latencies are estimates; they
# must be replaced with mixup-bench measurements before any real performance claims.
TESTBED: dict[str, DeviceSpec] = {
    "laptop-5060": DeviceSpec(
        execution=ExecutionType.GPU,
        cpu_cores=20,
        cpu_gflops=200.0,
        gpu_tflops=20.0,
        expert_fixed_ms=0.2,
        expert_per_ms=0.08,
        ram_bytes=30 * GB,
        vram_bytes=8 * GB,
        disk_read_mbps=2000.0,
        gpu_cpu_bw_mbps=16000.0,
        lead_score=5,
    ),
    "laptop-3070": DeviceSpec(
        execution=ExecutionType.GPU,
        cpu_cores=16,
        cpu_gflops=160.0,
        gpu_tflops=15.0,
        expert_fixed_ms=0.25,
        expert_per_ms=0.1,
        ram_bytes=16 * GB,
        vram_bytes=8 * GB,
        disk_read_mbps=2000.0,
        gpu_cpu_bw_mbps=16000.0,
        lead_score=4,
    ),
    "laptop-3060": DeviceSpec(
        execution=ExecutionType.GPU,
        cpu_cores=12,
        cpu_gflops=120.0,
        gpu_tflops=10.0,
        expert_fixed_ms=0.3,
        expert_per_ms=0.15,
        ram_bytes=16 * GB,
        vram_bytes=6 * GB,
        disk_read_mbps=1500.0,
        gpu_cpu_bw_mbps=16000.0,
        lead_score=3,
    ),
    "pi-1": DeviceSpec(
        execution=ExecutionType.CPU,
        cpu_cores=4,
        cpu_gflops=30.0,
        expert_fixed_ms=2.0,
        expert_per_ms=3.0,
        ram_bytes=8 * GB,
        disk_read_mbps=80.0,
        lead_score=1,
    ),
    "pi-2": DeviceSpec(
        execution=ExecutionType.CPU,
        cpu_cores=4,
        cpu_gflops=30.0,
        expert_fixed_ms=2.0,
        expert_per_ms=3.0,
        ram_bytes=8 * GB,
        disk_read_mbps=80.0,
        lead_score=1,
    ),
}


def build_testbed_profiles() -> dict[str, DeviceProfile]:
    """The placeholder five-device profiles (see TESTBED caveat)."""
    return {name: spec.to_profile(name) for name, spec in TESTBED.items()}
