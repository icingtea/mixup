"""``mixup-bench``: measure a device's compute/memory/disk and expert-serving latency.

Produces a ``DeviceProfile`` JSON. Run inside a node container (it needs torch and the
model file). The expert microbenchmark times the real ``GraniteMoeExperts`` forward for
1..k experts and fits ``expert_fixed_ms + n * expert_per_ms``, which is what the placement
cost model consumes.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import GraniteMoeConfig

from mixup_core.constants import MODEL_DIR, MODEL_FILE
from mixup_core.devices import DeviceProfile, ExecutionType

from .model import empty_experts

HIDDEN = 1024


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _matmul_gflops(device: torch.device, dtype: torch.dtype, n: int = 1024) -> float:
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    for _ in range(5):
        torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 2 * n * n * n * iters / elapsed / 1e9


def _ram_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def _disk_read_mbps(model_file: Path) -> float:
    size = model_file.stat().st_size
    start = time.perf_counter()
    with open(model_file, "rb") as f:
        while f.read(16 << 20):
            pass
    elapsed = time.perf_counter() - start
    return (size / (elapsed * 1_000_000.0)) if elapsed > 0 else 0.0


def _expert_latency_ms(
    config: GraniteMoeConfig, model_file: Path, device: torch.device
) -> tuple[float, float]:
    """Fit ``(fixed_ms, per_expert_ms)`` from a 1..k expert microbenchmark."""
    num_experts = 8
    experts = empty_experts(config, num_experts)
    with safe_open(model_file, framework="pt") as f:
        prefix = "model.layers.0.block_sparse_moe"
        gate_up = f.get_slice(f"{prefix}.input_linear.weight")[:num_experts]
        down = f.get_slice(f"{prefix}.output_linear.weight")[:num_experts]
    experts.load_state_dict({"gate_up_proj": gate_up, "down_proj": down}, assign=True)
    experts = experts.to(device).eval()

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    hidden = torch.randn(1, HIDDEN, dtype=dtype, device=device)
    weights = torch.ones(1, num_experts, dtype=dtype, device=device)

    times: dict[int, float] = {}
    for k in (1, 2, 4, 8):
        idx = torch.arange(k, device=device).unsqueeze(0)
        w = weights[:, :k]
        # warmup
        for _ in range(3):
            experts(hidden, idx, w)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reps = 50
        start = time.perf_counter()
        for _ in range(reps):
            experts(hidden, idx, w)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times[k] = (time.perf_counter() - start) / reps * 1_000.0

    # Least-squares fit of t(k) = fixed + per * k over the measured points.
    ks = list(times)
    n = len(ks)
    sx = sum(ks)
    sy = sum(times[k] for k in ks)
    sxx = sum(k * k for k in ks)
    sxy = sum(k * times[k] for k in ks)
    denom = n * sxx - sx * sx
    per = (n * sxy - sx * sy) / denom if denom else 0.0
    fixed = (sy - per * sx) / n if n else 0.0
    return max(0.0, fixed), max(0.0, per)


def profile(node_id: str, model_dir: Path = MODEL_DIR) -> DeviceProfile:
    device = _device()
    config = GraniteMoeConfig.from_pretrained(model_dir)
    fixed, per = _expert_latency_ms(config, MODEL_FILE, device)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return DeviceProfile(
            node_id=node_id,
            execution=ExecutionType.GPU,
            cpu_cores=os.cpu_count() or 1,
            cpu_gflops=_matmul_gflops(torch.device("cpu"), torch.float32, 512),
            gpu_tflops=_matmul_gflops(device, torch.bfloat16, 1024) / 1e3,
            expert_fixed_ms=fixed,
            expert_per_ms=per,
            ram_bytes=_ram_bytes(),
            vram_bytes=props.total_memory,
            disk_read_mbps=_disk_read_mbps(MODEL_FILE),
        )

    return DeviceProfile(
        node_id=node_id,
        execution=ExecutionType.CPU,
        cpu_cores=os.cpu_count() or 1,
        cpu_gflops=_matmul_gflops(device, torch.float32, 512),
        expert_fixed_ms=fixed,
        expert_per_ms=per,
        ram_bytes=_ram_bytes(),
        disk_read_mbps=_disk_read_mbps(MODEL_FILE),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a device profile.")
    parser.add_argument("--node", default="local", help="node id for the emitted profile")
    args = parser.parse_args(argv)
    print(json.dumps(profile(args.node).model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
