"""Live resource monitoring: sample CPU/memory/GPU and expose smoothed values.

Feeds the resource-perception layer (``mixup_core.resource``): raw measurements are
smoothed with EWMA and a rolling stability score, matching the report's §2.2.2. No psutil
dependency — CPU and memory come from ``/proc``, GPU from torch.
"""

import threading

import torch

from mixup_core.resource import MetricTracker

_KB = 1024


def _cpu_times() -> tuple[int, int]:
    """(total jiffies, idle jiffies) across all cores."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    values = [int(x) for x in parts[1:]]
    return sum(values), values[3]  # idle is the 4th field (index 3)


def _mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * _KB
    return 0


def _gpu_mem() -> tuple[int, int] | None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return total, free
    return None


class ResourceMonitor:
    """Samples CPU utilisation / free memory / GPU memory on a background thread and keeps
    EWMA-smoothed values + stability scores."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.cpu = MetricTracker(alpha=0.3, window=20)
        self.mem = MetricTracker(alpha=0.3, window=20)
        self._last_total, self._last_idle = _cpu_times()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            total, idle = _cpu_times()
            d_total = total - self._last_total
            d_idle = idle - self._last_idle
            self._last_total, self._last_idle = total, idle
            if d_total > 0:
                self.cpu.update(1.0 - d_idle / d_total)
            self.mem.update(_mem_available())

    def snapshot(self) -> dict:
        """Current smoothed resource state (plus raw GPU/memory figures)."""
        cpu_smoothed, cpu_stability = (
            (self.cpu.smoothed, _stability(self.cpu))
            if self.cpu.smoothed is not None
            else (0.0, 1.0)
        )
        mem_smoothed, mem_stability = (
            (self.mem.smoothed, _stability(self.mem))
            if self.mem.smoothed is not None
            else (0.0, 1.0)
        )
        gpu = _gpu_mem()
        return {
            "cpu_util": round(cpu_smoothed, 4),
            "cpu_util_stability": round(cpu_stability, 4),
            "mem_available_bytes": int(mem_smoothed),
            "mem_stability": round(mem_stability, 4),
            "gpu_total_bytes": gpu[0] if gpu else None,
            "gpu_free_bytes": gpu[1] if gpu else None,
        }


def _stability(tracker: MetricTracker) -> float:
    from mixup_core.resource import stability

    return stability(tracker.recent, tracker.smoothed or 0.0)
