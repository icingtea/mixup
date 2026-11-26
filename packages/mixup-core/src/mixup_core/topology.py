"""Expert routing: which container holds which expert, and its local slot there.

The layout is derived from StartupConfig.experts (the explicit per-node lists), not
recomputed with a ``//``/``%`` heuristic, so the routing table can never disagree
with the config.
"""

from .runtime import ClusterView, MemberState, RoutingEntry, RoutingTable
from .types import ExpertId, ExpertRoute, StartupConfig


def build_lut(config: StartupConfig) -> dict[ExpertId, ExpertRoute]:
    """Global expert id -> (container holding it, its slot on that container)."""
    lut: dict[ExpertId, ExpertRoute] = {}
    for container, experts in config.experts.items():
        for slot, expert in enumerate(experts):
            lut[expert] = (container, slot)
    return lut


def expert_containers(config: StartupConfig) -> list[str]:
    """The expert containers, in config order."""
    return list(config.experts)


def experts_for(lut: dict[ExpertId, ExpertRoute], container: str) -> list[ExpertId]:
    """The (sorted) experts held by `container`."""
    return sorted(e for e, (c, _) in lut.items() if c == container)


def validate_partition(config: StartupConfig, num_experts: int) -> None:
    """Raise unless config.experts is a contiguous, complete partition of 0..num_experts-1.

    Complete coverage matters because ExpertGroup.forward indexes the LUT over every
    expert id. Contiguity matters because ExpertGroup loads each container's experts
    as one consecutive slice of the big safetensors file.
    """
    lut = build_lut(config)
    covered = set(lut)
    expected = set(range(num_experts))

    if covered != expected:
        raise ValueError(
            f"expert allocation does not cover 0..{num_experts - 1}: "
            f"missing={sorted(expected - covered)} extra={sorted(covered - expected)}"
        )

    for container, experts in config.experts.items():
        ordered = sorted(experts)
        if ordered != list(range(min(ordered), max(ordered) + 1)):
            raise ValueError(f"container {container!r} experts are not contiguous: {experts}")


def routing_table_from_config(config: StartupConfig) -> RoutingTable:
    """Build the initial epoch-0 routing table from the startup layout."""
    lut = build_lut(config)
    assignment = {
        expert: RoutingEntry(node_id=node, slot=slot)
        for expert, (node, slot) in lut.items()
    }
    return RoutingTable(epoch=0, assignment=assignment)


def cluster_view_from_config(config: StartupConfig) -> ClusterView:
    """Build the initial live view (members + routing) from the startup layout.

    Priorities are derived from roster order (earlier = higher), so the initial leader
    is the highest-priority member.
    """
    n = len(config.nodes)
    members = {
        name: MemberState(
            node_id=name,
            role=config.initial_role(name),
            epoch=0,
            incarnation=1,
            priority=n - index,
            owned_experts=list(config.experts.get(name, [])),
        )
        for index, name in enumerate(config.nodes)
    }
    return ClusterView(members=members, routing=routing_table_from_config(config))
