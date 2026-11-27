"""Host-side mapping from node names to their externally-reachable HTTP URLs."""

from .constants import EXTERNAL_PORT_BASE
from .types import StartupConfig


def node_urls(config: StartupConfig) -> dict[str, str]:
    """Map each node to its host URL, mirroring the orchestrator's port assignment.

    The leader is published on ``EXTERNAL_PORT_BASE`` and every other node (in roster
    order) on ``EXTERNAL_PORT_BASE + 1 + offset``. Using the full ``nodes`` roster (rather
    than just ``experts``) keeps the mapping correct when the leader also owns experts
    (leader-as-owner) and matches ``mixup-lab``. The REPL and the gateway both use this.
    """
    urls = {config.initial_leader: f"http://localhost:{EXTERNAL_PORT_BASE}"}
    others = [n for n in config.nodes if n != config.initial_leader]
    for offset, name in enumerate(others):
        urls[name] = f"http://localhost:{EXTERNAL_PORT_BASE + offset + 1}"
    return urls
