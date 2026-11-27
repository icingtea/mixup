"""Deterministic trace-driven simulator.

Replays recorded activation traces through a placement + cost model and reports the
resulting latency distribution and per-node load. The single-stream model mirrors the
current dispatch loop exactly: tokens are sequential, layers within a token are sequential,
and the nodes serving a layer run in parallel (so a layer's time is the max over its
nodes). Queueing only appears when a node is still busy from an earlier request, which for
a single sequential stream is never — the machinery is in place for the concurrency model
(Phase 7), where multiple streams interleave.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .placement import LatencyStats, PlacementProblem, loads_for_layer, summarize_latencies
from .types import ExpertId, NodeId


@dataclass
class SimulationResult:
    """Outcome of replaying one stream of tokens."""

    stats: LatencyStats
    per_node_served: dict[NodeId, int] = field(default_factory=dict)
    per_node_compute_ms: dict[NodeId, float] = field(default_factory=dict)
    queue_ms: float = 0.0


class Simulator:
    """Replays a placement against the activation trace with per-node busy tracking."""

    def __init__(self, problem: PlacementProblem) -> None:
        self.problem = problem

    def run_tokens(
        self,
        assignment: dict[ExpertId, NodeId],
        tokens: list[list[list[ExpertId]]],
        queueing: bool = True,
    ) -> SimulationResult:
        """Replay ``tokens`` (each ``tokens[t][l]`` = selected experts at layer ``l``)."""
        busy: dict[NodeId, float] = {n: 0.0 for n in self.problem.nodes()}
        served: Counter[NodeId] = Counter()
        compute: dict[NodeId, float] = defaultdict(float)
        queue_total = 0.0
        latencies: list[float] = []

        now = 0.0  # global clock: one continuous generation stream
        for token in tokens:
            token_time = 0.0
            for layer_experts in token:
                loads = loads_for_layer(assignment, layer_experts)
                layer_time = 0.0
                for node, n in loads.items():
                    if n <= 0:
                        continue
                    rtt = self.problem.rtt_ms.get(node, 0.0)
                    service = self.problem.cost.compute_ms(node, n) + rtt
                    start = max(now, busy[node]) if queueing else now
                    queue_total += start - now
                    end = start + service
                    busy[node] = end
                    served[node] += n
                    compute[node] += service
                    layer_time = max(layer_time, end - now)
                token_time += layer_time
                now += layer_time
            now += self.problem.dense_ms
            latencies.append(token_time + self.problem.dense_ms)

        return SimulationResult(
            stats=summarize_latencies(latencies),
            per_node_served=dict(served),
            per_node_compute_ms=dict(compute),
            queue_ms=queue_total,
        )

    def run(
        self,
        assignment: dict[ExpertId, NodeId],
        num_samples: int = 64,
        seed: int = 0,
        queueing: bool = True,
    ) -> SimulationResult:
        tokens = self.problem.stats.sample_tokens(num_samples, seed)
        return self.run_tokens(assignment, tokens, queueing=queueing)
