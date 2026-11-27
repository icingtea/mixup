"""Tests for storage tiers and the residency/offload decisions."""

import pytest

from mixup_core.devices import DeviceProfile
from mixup_core.residency import (
    Residency,
    eviction_score,
    prefetch_threshold,
    retention_score,
)
from mixup_core.tiers import ResidencyState, Tier


def _gpu(**kw) -> DeviceProfile:
    d = dict(node_id="g", execution="gpu", disk_read_mbps=1000.0, gpu_cpu_bw_mbps=10000.0)
    d.update(kw)
    return DeviceProfile(**d)


def _cpu(**kw) -> DeviceProfile:
    d = dict(node_id="c", execution="cpu", disk_read_mbps=500.0)
    d.update(kw)
    return DeviceProfile(**d)


# -- Tier -----------------------------------------------------------------


def test_ladder_gpu() -> None:
    assert Tier.for_device(_gpu()) == [Tier.GPU, Tier.CPU, Tier.DISK]


def test_ladder_cpu() -> None:
    assert Tier.for_device(_cpu()) == [Tier.CPU, Tier.DISK]


def test_can_execute() -> None:
    assert Tier.can_execute(Tier.GPU, _gpu())
    assert not Tier.can_execute(Tier.CPU, _gpu())
    assert not Tier.can_execute(Tier.DISK, _gpu())
    assert Tier.can_execute(Tier.CPU, _cpu())
    assert not Tier.can_execute(Tier.DISK, _cpu())


def test_state_touch_decays_frequency() -> None:
    s = ResidencyState(tier=Tier.CPU, bytes=10)
    s.touch(1, 0.9)
    s.touch(2, 0.9)
    assert s.use_count == pytest.approx(1.0 + 0.9 * 1.0)  # 1.9
    assert s.last_used == 2


# -- scores ---------------------------------------------------------------


def test_retention_higher_frequency_wins() -> None:
    assert retention_score(0.9, 10.0, 0.6) > retention_score(0.1, 10.0, 0.6)


def test_retention_higher_load_wins() -> None:
    a = retention_score(0.5, 50.0, 0.4, load_scale=100.0)
    b = retention_score(0.5, 5.0, 0.4, load_scale=100.0)
    assert a > b


def test_eviction_lower_probability_evicts_first() -> None:
    assert eviction_score(0.1, 1.0, 1.0, 0.5, 0.5) > eviction_score(0.9, 1.0, 1.0, 0.5, 0.5)


def test_eviction_less_recent_evicts_first() -> None:
    # recency = 1/(age+1); older (smaller recency) -> higher score.
    assert eviction_score(0.5, 0.1, 1.0, 0.5, 0.5) > eviction_score(0.5, 1.0, 1.0, 0.5, 0.5)


def test_eviction_less_important_evicts_first() -> None:
    assert eviction_score(0.5, 1.0, 0.1, 0.5, 0.5) > eviction_score(0.5, 1.0, 1.0, 0.5, 0.5)


def test_eviction_clamps_zero_inverses() -> None:
    # importance and recency both ~0 must not divide by zero.
    assert eviction_score(0.0, 0.0, 0.0, 0.5, 0.5) > 0


def test_prefetch_threshold_more_free_lowers() -> None:
    assert prefetch_threshold(0.05, 1.0, 0.8) < prefetch_threshold(0.05, 1.0, 0.0)


def test_prefetch_threshold_higher_stability_lowers() -> None:
    assert prefetch_threshold(0.05, 1.0, 0.5) < prefetch_threshold(0.05, 0.2, 0.5)


def test_prefetch_threshold_stability_floor_avoids_zero() -> None:
    assert prefetch_threshold(0.05, 0.0, 0.0) > 0


# -- Residency manager ----------------------------------------------------


def _res(
    profile: DeviceProfile, budget: dict[str, int] | None = None, bytes_: int = 100
) -> Residency:
    return Residency(
        profile=profile,
        expert_bytes_per_layer=bytes_,
        budget_bytes=budget or {},
    )


def test_used_and_free_bytes() -> None:
    r = _res(_cpu(), {"cpu": 1000})
    r.place(0, 0, Tier.CPU, 300)
    r.place(0, 1, Tier.CPU, 200)
    assert r.used_bytes(Tier.CPU) == 500
    assert r.free_bytes(Tier.CPU) == 500
    assert r.free_ratio(Tier.CPU) == pytest.approx(0.5)


def test_place_moves_existing_key() -> None:
    r = _res(_cpu(), {"cpu": 1000})
    r.place(0, 0, Tier.CPU, 300)
    r.place(0, 0, Tier.DISK, 300)
    assert r.states[(0, 0)].tier == Tier.DISK
    assert r.used_bytes(Tier.CPU) == 0
    assert r.used_bytes(Tier.DISK) == 300


def test_eviction_order_cold_first() -> None:
    r = _res(_cpu())
    for e in range(4):
        r.place(0, e, Tier.CPU, 100)
    r.touch(0, 3)  # expert 3 is the most recent/hot
    order = r.eviction_order(Tier.CPU, {(0, 3): 0.9, (0, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0})
    # expert 3 (hot + recent) is evicted last.
    assert order[-1] == (0, 3)
    assert (0, 3) not in order[:-1]


def test_eviction_respects_pinned() -> None:
    r = _res(_cpu())
    for e in range(3):
        r.place(0, e, Tier.CPU, 100)
    r.states[(0, 1)].pinned = True
    order = r.eviction_order(Tier.CPU, {(0, e): 0.0 for e in range(3)})
    assert (0, 1) not in order


def test_ensure_room_evicts_until_fit() -> None:
    r = _res(_cpu(), {"cpu": 300})
    r.place(0, 0, Tier.CPU, 100)
    r.place(0, 1, Tier.CPU, 100)
    r.place(0, 2, Tier.CPU, 100)  # exactly full (300)
    # Need 100 more -> evict one expert.
    to_evict = r.ensure_room(Tier.CPU, 100, {(0, e): 0.0 for e in range(3)})
    assert len(to_evict) == 1


def test_ensure_room_noop_when_room() -> None:
    r = _res(_cpu(), {"cpu": 500})
    r.place(0, 0, Tier.CPU, 100)
    assert r.ensure_room(Tier.CPU, 100, {}) == []


def test_ensure_room_unbounded_budget_never_evicts() -> None:
    r = _res(_cpu())  # no budget -> unbounded
    r.place(0, 0, Tier.CPU, 100)
    assert r.ensure_room(Tier.CPU, 1_000_000, {}) == []
