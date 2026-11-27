"""``mixup-place``: evaluate placement policies on a recorded activation trace.

Pure-core evaluation of the Phase 2 placement engine: loads a ``mixup-trace`` JSONL, builds
a heterogeneous device pool, and reports p50/p95/mean per-token expert-dispatch latency for
the uniform, frequency, and latency-aware policies. This is the simulation-side of the
"resource-aware placement beats uniform" gate.
"""

import argparse
import json

from .placement import FrequencyPolicy, LatencyPolicy, PlacementProblem, UniformPolicy, evaluate
from .profiles import DeviceSpec
from .stats import load_trace_file


def _profiles_from_names(names: list[str]) -> dict:
    specs = {
        "gpu-fast": DeviceSpec(execution="gpu", expert_fixed_ms=0.2, expert_per_ms=0.08),
        "gpu-mid": DeviceSpec(execution="gpu", expert_fixed_ms=0.3, expert_per_ms=0.15),
        "cpu": DeviceSpec(execution="cpu", expert_fixed_ms=2.0, expert_per_ms=3.0),
    }
    profiles: dict[str, object] = {}
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        node_id = name if counts[name] == 1 else f"{name}-{counts[name]}"
        profiles[node_id] = specs[name].to_profile(node_id)
    return profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate placement policies on a trace.")
    parser.add_argument("trace", help="mixup-trace JSONL file")
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["gpu-fast", "cpu", "cpu"],
        help="device pool as spec names: gpu-fast|gpu-mid|cpu (repeatable)",
    )
    parser.add_argument("--rtt-ms", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-ms", type=float, default=0.0)
    parser.add_argument(
        "--budget-mb",
        type=float,
        default=None,
        help="per-node expert budget in MB (constrains how many experts each node holds)",
    )
    parser.add_argument(
        "--expert-mb",
        type=float,
        default=75.5,
        help="bytes of one expert across all layers, in MB (Granite 1B-A400M ~75.5)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    stats = load_trace_file(args.trace)
    profiles = _profiles_from_names(args.nodes)
    budget_bytes = (
        {n: int(args.budget_mb * 1_000_000) for n in profiles}
        if args.budget_mb is not None
        else None
    )
    problem = PlacementProblem(
        num_experts=stats.num_experts,
        profiles=profiles,
        stats=stats,
        rtt_ms={n: args.rtt_ms for n in profiles},
        leader=args.nodes[0],
        dense_ms=args.dense_ms,
        expert_bytes=int(args.expert_mb * 1_000_000),
        budget_bytes=budget_bytes,
    )

    results = {}
    for name, policy in (
        ("uniform", UniformPolicy()),
        ("frequency", FrequencyPolicy()),
        ("latency", LatencyPolicy(max_iters=20, num_samples=args.num_samples, seed=args.seed)),
    ):
        assignment = policy.place(problem)
        lat = evaluate(problem, assignment, num_samples=args.num_samples, seed=args.seed)
        results[name] = {
            "p50_ms": round(lat.p50_ms, 3),
            "p95_ms": round(lat.p95_ms, 3),
            "mean_ms": round(lat.mean_ms, 3),
            "assignment": assignment,
        }

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for name, r in results.items():
        print(
            f"{name:<10} p50={r['p50_ms']:>8.3f}ms  p95={r['p95_ms']:>8.3f}ms  "
            f"mean={r['mean_ms']:>8.3f}ms"
        )
    print("\nlatency assignment:")
    for e, node in sorted(results["latency"]["assignment"].items()):
        print(f"  expert {e:>2} -> {node}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
