"""Tests for the placement planner (perception -> routing table)."""

from mixup_core.planner import plan_routing
from mixup_core.profiles import DeviceSpec
from mixup_core.runtime import RoutingTable
from mixup_core.tracegen import generate_trace, trace_to_stats


def _hetero(num_experts: int = 32, num_layers: int = 24, tokens: int = 150, seed: int = 0):
    trace = generate_trace(num_layers, num_experts, 8, tokens, zipf=1.0, seed=seed)
    stats = trace_to_stats(trace, num_layers, num_experts)
    specs = {
        "gpu": DeviceSpec(execution="gpu", expert_fixed_ms=0.2, expert_per_ms=0.08),
        "pi1": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
        "pi2": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
    }
    profiles = {name: spec.to_profile(name) for name, spec in specs.items()}
    return profiles, stats


def test_plan_routing_covers_all_experts() -> None:
    profiles, stats = _hetero()
    table = plan_routing(profiles, stats, {"gpu": 0.5, "pi1": 0.5, "pi2": 0.5}, "gpu")
    assert set(table.assignment) == set(range(32))


def test_plan_routing_epoch_starts_at_one() -> None:
    profiles, stats = _hetero()
    table = plan_routing(profiles, stats, {}, "gpu")
    assert table.epoch == 1


def test_plan_routing_bumps_epoch() -> None:
    profiles, stats = _hetero()
    prev = RoutingTable(epoch=7, assignment={})
    table = plan_routing(profiles, stats, {}, "gpu", prev_table=prev)
    assert table.epoch == 8


def test_plan_routing_is_deterministic() -> None:
    profiles, stats = _hetero()
    rtt = {"gpu": 0.5, "pi1": 0.5, "pi2": 0.5}
    assert plan_routing(profiles, stats, rtt, "gpu") == plan_routing(profiles, stats, rtt, "gpu")


def test_plan_routing_respects_budget() -> None:
    profiles, stats = _hetero()
    # 12 experts per node max (1 MB expert), 36 capacity >= 32.
    budgets = {n: 12 * 1_000_000 for n in profiles}
    table = plan_routing(profiles, stats, {}, "gpu", budget_bytes=budgets, expert_bytes=1_000_000)
    owners = list(table.assignment.values())
    for node in profiles:
        assert sum(1 for e in owners if e.node_id == node) <= 12


def test_plan_routing_slots_are_contiguous_per_node() -> None:
    profiles, stats = _hetero()
    table = plan_routing(profiles, stats, {}, "gpu")
    by_node: dict[str, list[int]] = {}
    for e, entry in table.assignment.items():
        by_node.setdefault(entry.node_id, []).append((entry.slot, e))
    for node, pairs in by_node.items():
        slots = sorted(s for s, _ in pairs)
        assert slots == list(range(len(slots))), f"non-contiguous slots on {node}"


def test_uniform_policy_is_available() -> None:
    profiles, stats = _hetero()
    table = plan_routing(profiles, stats, {}, "gpu", policy="uniform")
    assert set(table.assignment) == set(range(32))
