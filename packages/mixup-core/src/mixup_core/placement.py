"""Expert placement: the decision layer.

``PlacementProblem`` bundles the profiles, cost model, activation statistics, and network
state into one value; policies turn it into an ``expert -> node`` assignment. The objective
is the *expected tail latency* (p95 over sampled tokens) plus a churn penalty for moving
experts off their current owner and a hard memory-budget penalty.

Everything here is pure and deterministic (given ``stats`` and a ``seed``), so policies are
unit-testable and the simulator can replay them.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

from .cost import CostModel
from .devices import DeviceProfile
from .stats import ActivationStats
from .types import ExpertId, NodeId

MEMORY_PENALTY_MS_PER_BYTE = 1e-5
"""Penalty rate for exceeding a node's expert budget: 75 MB over → 750 ms."""

DEFAULT_EXPERT_BYTES = 75_500_000
"""Bytes of one expert across all layers for Granite 1B-A400M (~3.15 MB/layer x 24)."""


def _percentile(values: list[float], q: float) -> float:
    """``q``-th percentile (0..1) of ``values`` (sorted copy), nearest-rank."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


@dataclass
class LatencyStats:
    """Latency distribution of a placement over sampled tokens."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float
    num_samples: int


@dataclass
class PlacementAnalysis:
    """Full evaluation of a placement: latency stats + churn + memory violation."""

    stats: LatencyStats
    churn_ms: float
    violation_bytes: int
    objective_ms: float


@dataclass
class PlacementProblem:
    """Everything a policy needs to choose an ``expert -> node`` assignment."""

    num_experts: int
    profiles: dict[NodeId, DeviceProfile]
    stats: ActivationStats
    rtt_ms: dict[NodeId, float]
    leader: NodeId
    expert_bytes: int = DEFAULT_EXPERT_BYTES
    dense_ms: float = 0.0
    prev: dict[ExpertId, NodeId] | None = None
    churn_weight: float = 1.0
    budget_bytes: dict[NodeId, int] | None = None

    cost: CostModel = field(init=False)

    def __post_init__(self) -> None:
        self.cost = CostModel(self.profiles)

    def nodes(self) -> list[NodeId]:
        return sorted(self.profiles)

    def capacity(self, node: NodeId) -> int:
        if self.budget_bytes is None or node not in self.budget_bytes:
            return self.num_experts
        return max(0, self.budget_bytes[node] // self.expert_bytes)


def activation_mass(stats: ActivationStats) -> dict[ExpertId, float]:
    """Total expected selections per token, summed over layers, per expert."""
    marginals = stats.marginals()
    return {
        e: sum(marginals[layer][e] for layer in range(stats.num_layers))
        for e in range(stats.num_experts)
    }


def loads_for_layer(
    assignment: dict[ExpertId, NodeId], layer_experts: list[ExpertId]
) -> dict[NodeId, int]:
    """How many experts each node serves for one layer's selections."""
    loads: dict[NodeId, int] = {}
    for e in layer_experts:
        node = assignment[e]
        loads[node] = loads.get(node, 0) + 1
    return loads


def evaluate(
    problem: PlacementProblem,
    assignment: dict[ExpertId, NodeId],
    num_samples: int = 64,
    seed: int = 0,
) -> LatencyStats:
    """Expected token latency over ``num_samples`` sampled tokens."""
    tokens = problem.stats.sample_tokens(num_samples, seed)
    latencies: list[float] = []
    for token in tokens:
        total = 0.0
        for layer_experts in token:
            total += problem.cost.layer_ms(
                loads_for_layer(assignment, layer_experts), problem.rtt_ms
            )
        latencies.append(total + problem.dense_ms)

    if not latencies:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0)
    return LatencyStats(
        p50_ms=_percentile(latencies, 0.5),
        p95_ms=_percentile(latencies, 0.95),
        mean_ms=sum(latencies) / len(latencies),
        max_ms=max(latencies),
        num_samples=len(latencies),
    )


def moved_experts(problem: PlacementProblem, assignment: dict[ExpertId, NodeId]) -> list[ExpertId]:
    """Experts whose owner changed relative to ``problem.prev``."""
    if problem.prev is None:
        return []
    return [e for e, node in assignment.items() if problem.prev.get(e) != node]


def churn_ms(problem: PlacementProblem, assignment: dict[ExpertId, NodeId]) -> float:
    """Cost of moving experts off their current owner: load time on the new owner."""
    total = 0.0
    for e in moved_experts(problem, assignment):
        new_owner = assignment[e]
        profile = problem.profiles[new_owner]
        if profile.disk_read_mbps > 0:
            total += problem.expert_bytes / (profile.disk_read_mbps * 1_000_000.0) * 1_000.0
    return total


def violation_bytes(problem: PlacementProblem, assignment: dict[ExpertId, NodeId]) -> int:
    """Bytes by which any node exceeds its expert budget (summed across nodes)."""
    if problem.budget_bytes is None:
        return 0
    by_node: dict[NodeId, int] = {}
    for _e, node in assignment.items():
        by_node[node] = by_node.get(node, 0) + problem.expert_bytes
    return sum(
        max(0, used - problem.budget_bytes[node])
        for node, used in by_node.items()
        if node in problem.budget_bytes
    )


def objective(
    problem: PlacementProblem,
    assignment: dict[ExpertId, NodeId],
    num_samples: int = 64,
    seed: int = 0,
) -> float:
    """Minimize this: p95 latency + churn + memory-violation penalty."""
    stats = evaluate(problem, assignment, num_samples, seed)
    return (
        stats.p95_ms
        + problem.churn_weight * churn_ms(problem, assignment)
        + MEMORY_PENALTY_MS_PER_BYTE * violation_bytes(problem, assignment)
    )


def analyze(
    problem: PlacementProblem,
    assignment: dict[ExpertId, NodeId],
    num_samples: int = 64,
    seed: int = 0,
) -> PlacementAnalysis:
    stats = evaluate(problem, assignment, num_samples, seed)
    churn = churn_ms(problem, assignment)
    violation = violation_bytes(problem, assignment)
    return PlacementAnalysis(
        stats=stats,
        churn_ms=churn,
        violation_bytes=violation,
        objective_ms=stats.p95_ms
        + problem.churn_weight * churn
        + MEMORY_PENALTY_MS_PER_BYTE * violation,
    )


class PlacementPolicy(Protocol):
    def place(self, problem: PlacementProblem) -> dict[ExpertId, NodeId]: ...


class UniformPolicy:
    """Round-robin experts across nodes (capacity-aware), ignoring activation frequency."""

    def place(self, problem: PlacementProblem) -> dict[ExpertId, NodeId]:
        nodes = problem.nodes()
        cap = {n: problem.capacity(n) for n in nodes}
        assignment: dict[ExpertId, NodeId] = {}
        i = 0
        for e in range(problem.num_experts):
            # Find the next node with remaining capacity (round-robin, deterministic).
            for _ in range(len(nodes)):
                node = nodes[i % len(nodes)]
                i += 1
                if cap[node] > 0:
                    assignment[e] = node
                    cap[node] -= 1
                    break
        return assignment


class FrequencyPolicy:
    """Greedy: hottest experts onto the fastest nodes, balancing expected served load."""

    def place(self, problem: PlacementProblem) -> dict[ExpertId, NodeId]:
        nodes = problem.nodes()
        cap = {n: problem.capacity(n) for n in nodes}
        masses = activation_mass(problem.stats)
        order = sorted(range(problem.num_experts), key=lambda e: (-masses[e], e))

        load = {n: 0.0 for n in nodes}  # expected selections served per token
        assignment: dict[ExpertId, NodeId] = {}
        for e in order:
            candidates = [n for n in nodes if cap[n] > 0]
            if not candidates:
                raise ValueError("placement infeasible: no node has remaining capacity")
            node = min(
                candidates,
                key=lambda n: (problem.profiles[n].expert_per_ms * (load[n] + masses[e]), n),
            )
            assignment[e] = node
            load[node] += masses[e]
            cap[node] -= 1
        return assignment


class LatencyPolicy:
    """Hill-climb from a frequency-greedy seed, minimizing the sampled tail-latency objective.

    Moves one expert at a time to a different node while the objective strictly improves.
    Deterministic: fixed seed, fixed node/order iteration. This is the resource-aware
    placement policy of Phase 2.
    """

    def __init__(self, max_iters: int = 50, num_samples: int = 32, seed: int = 0) -> None:
        self.max_iters = max_iters
        self.num_samples = num_samples
        self.seed = seed

    def place(self, problem: PlacementProblem) -> dict[ExpertId, NodeId]:
        best = FrequencyPolicy().place(problem)
        best_score = objective(problem, best, self.num_samples, self.seed)

        for _ in range(self.max_iters):
            improved = False
            for e in range(problem.num_experts):
                for node in problem.nodes():
                    if best[e] == node:
                        continue
                    candidate = dict(best)
                    candidate[e] = node
                    score = objective(problem, candidate, self.num_samples, self.seed)
                    if score < best_score:
                        best, best_score = candidate, score
                        improved = True
            if not improved:
                break
        return best
