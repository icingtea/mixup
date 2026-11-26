"""Host-side mapping from node names to their externally-reachable HTTP URLs."""

from .constants import EXTERNAL_PORT_BASE
from .types import StartupConfig


def node_urls(config: StartupConfig) -> dict[str, str]:
    """Map each node to its host URL, mirroring the orchestrator's port assignment.

    The orchestrator publishes the leader on ``EXTERNAL_PORT_BASE`` and each expert on
    ``EXTERNAL_PORT_BASE + 1 + offset``, where ``offset`` is the expert's position in
    ``config.experts`` (insertion order). The REPL and the gateway both use this, so it
    lives here next to the constants it mirrors.
    """
    urls = {config.initial_leader: f"http://localhost:{EXTERNAL_PORT_BASE}"}
    for offset, name in enumerate(config.experts):
        urls[name] = f"http://localhost:{EXTERNAL_PORT_BASE + offset + 1}"
    return urls
