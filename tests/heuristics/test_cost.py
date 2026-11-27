"""Tests for the latency cost model."""

import pytest

from mixup_core.cost import CostModel
from mixup_core.devices import DeviceProfile


def _gpu(node: str = "g", **kw) -> DeviceProfile:
    defaults = dict(
        execution="gpu",
        expert_fixed_ms=1.0,
        expert_per_ms=0.5,
        gpu_cpu_bw_mbps=10_000.0,
        disk_read_mbps=1_000.0,
    )
    defaults.update(kw)
    return DeviceProfile(node_id=node, **defaults)


def _cpu(node: str = "c", **kw) -> DeviceProfile:
    defaults = dict(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=4.0, disk_read_mbps=500.0)
    defaults.update(kw)
    return DeviceProfile(node_id=node, **defaults)


def test_compute_is_fixed_plus_per_expert() -> None:
    model = CostModel({"g": _gpu()})
    assert model.compute_ms("g", 0) == pytest.approx(1.0)
    assert model.compute_ms("g", 3) == pytest.approx(1.0 + 3 * 0.5)


def test_network_is_rtt() -> None:
    model = CostModel({"g": _gpu()})
    assert model.network_ms(3.2) == pytest.approx(3.2)


def test_gpu_resident_is_free() -> None:
    model = CostModel({"g": _gpu()})
    assert model.load_ms("g", "gpu", 1_000_000) == 0.0


def test_gpu_from_cpu_uses_pcie() -> None:
    model = CostModel({"g": _gpu(gpu_cpu_bw_mbps=1_000.0)})
    # 1 MB over 1000 MB/s = 1 ms.
    assert model.load_ms("g", "cpu", 1_000_000) == pytest.approx(1.0)


def test_gpu_from_disk_is_two_stage() -> None:
    model = CostModel({"g": _gpu(gpu_cpu_bw_mbps=1_000.0, disk_read_mbps=500.0)})
    # 1 MB: disk (2 ms @ 500 MB/s) + PCIe (1 ms @ 1000 MB/s) = 3 ms.
    assert model.load_ms("g", "disk", 1_000_000) == pytest.approx(3.0)


def test_cpu_resident_is_free() -> None:
    model = CostModel({"c": _cpu()})
    assert model.load_ms("c", "cpu", 1_000_000) == 0.0


def test_cpu_from_disk_is_one_stage() -> None:
    model = CostModel({"c": _cpu(disk_read_mbps=500.0)})
    # 1 MB @ 500 MB/s = 2 ms.
    assert model.load_ms("c", "disk", 1_000_000) == pytest.approx(2.0)


def test_cpu_has_no_gpu_tier() -> None:
    model = CostModel({"c": _cpu()})
    with pytest.raises(ValueError):
        model.load_ms("c", "gpu", 1_000_000)


def test_gpu_missing_pcie_raises() -> None:
    model = CostModel({"g": _gpu(gpu_cpu_bw_mbps=None)})
    with pytest.raises(ValueError):
        model.load_ms("g", "cpu", 1_000_000)


def test_layer_ms_is_max_over_nodes() -> None:
    model = CostModel({"a": _gpu("a"), "b": _gpu("b", expert_per_ms=2.0)})
    loads = {"a": 2, "b": 1}
    rtt = {"a": 1.0, "b": 1.0}
    # a: 1 + 2*0.5 + 1 rtt = 3.0; b: 1 + 1*2 + 1 rtt = 4.0 -> max 4.0.
    assert model.layer_ms(loads, rtt) == pytest.approx(4.0)


def test_layer_ms_ignores_zero_load_nodes() -> None:
    model = CostModel({"a": _gpu("a"), "b": _gpu("b", expert_per_ms=100.0)})
    loads = {"a": 2, "b": 0}
    # b contributes nothing (its per_expert=100 would otherwise dominate).
    assert model.layer_ms(loads, {}) == pytest.approx(1.0 + 2 * 0.5)


def test_layer_ms_empty_is_zero() -> None:
    model = CostModel({"a": _gpu("a")})
    assert model.layer_ms({}, {}) == 0.0


def test_token_ms_sums_layers_and_dense() -> None:
    model = CostModel({"a": _gpu("a")})
    layers = [{"a": 1}, {"a": 2}]
    # dense 5 + (1 + 1*0.5) + (1 + 2*0.5) = 5 + 1.5 + 2.0 = 8.5.
    assert model.token_ms(layers, {}, dense_ms=5.0) == pytest.approx(8.5)
