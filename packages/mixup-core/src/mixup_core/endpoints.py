"""Resolve how to reach a node, for both single-host and cross-host deployments.

Single-host labs leave ``StartupConfig.endpoints`` empty; nodes then fall back to Docker
container names (``http://<name>:8000`` and ``(<name>, 8001)``). Cross-host deployments fill
in explicit endpoints, which both the nodes and the host-side tools (REPL, web) use.
"""

from .constants import EXPERT_PORT, address
from .types import StartupConfig


class EndpointResolver:
    """Map a node id to its HTTP control URL and its expert data-plane ``(host, port)``."""

    def __init__(self, config: StartupConfig) -> None:
        self._endpoints = config.endpoints

    def http(self, node: str) -> str:
        """The control-plane base URL for ``node`` (falling back to the Docker default)."""
        endpoint = self._endpoints.get(node)
        return endpoint.http if endpoint is not None else address(node)

    def expert(self, node: str) -> tuple[str, int]:
        """The expert data-plane ``(host, port)`` for ``node`` (Docker default fallback)."""
        endpoint = self._endpoints.get(node)
        if endpoint is not None:
            return endpoint.expert_host, endpoint.expert_port
        return node, EXPERT_PORT
