"""Smoke test: verify every expert is assigned to exactly one node, correctly.

Runs without torch or Docker. It checks the routing table derived from config.json,
which is exactly what decides which node loads which experts.
"""

import argparse
import sys

from .startup_config import load_startup_config
from .topology import build_lut, validate_partition

# granite-3.1-1b-a400m-instruct has 32 local experts per layer.
DEFAULT_NUM_EXPERTS = 32


def main(argv: list[str] | None = None) -> int:
    """Print the expert-to-node assignment and verify it is complete and contiguous."""
    parser = argparse.ArgumentParser(description="Verify the expert-to-node routing.")
    parser.add_argument(
        "config", nargs="?", default="startup_config.json", help="path to the cluster config"
    )
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    args = parser.parse_args(argv)

    config = load_startup_config(args.config)

    print(f"initial_leader: {config.initial_leader}")
    print("experts:")
    for name, experts in config.experts.items():
        if not experts:
            span = "[]"
        elif experts == list(range(min(experts), max(experts) + 1)):
            span = f"[{experts[0]}..{experts[-1]}]"
        else:
            span = str(experts)
        print(f"  {name}: {span}  ({len(experts)} experts)")

    try:
        validate_partition(config, args.num_experts)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    lut = build_lut(config)
    print(f"OK: {len(lut)}/{args.num_experts} experts covered, no overlaps, contiguous per node")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
