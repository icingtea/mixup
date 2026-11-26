"""Central-side HTTP client: fan expert work out to the containers and sum replies.

Resilient: on a connection failure to an expert, it invokes ``on_failure`` — which the
central uses to rebalance and broadcast the new routing table — then retries against the
new owner (Phase 6/7 failover).
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from requests.exceptions import RequestException
from torch import Tensor

from mixup_core.constants import address
from mixup_core.runtime import RoutingTable

from .serialization import ExpertRequest, ExpertResponse

logger = logging.getLogger(__name__)


class _NodeFailure(Exception):
    """Raised when an expert node is unreachable; carries the node name."""

    def __init__(self, node: str) -> None:
        super().__init__(f"node {node} unreachable")
        self.node = node


class Dispatcher:
    """Runs each token's chosen experts on the containers that hold them, in parallel."""

    def __init__(
        self,
        name: str,
        table: RoutingTable,
        on_failure: Callable[[str], RoutingTable],
        max_retries: int = 2,
    ) -> None:
        self.name = name
        self.table = table
        self.on_failure = on_failure  # (node) -> new RoutingTable (after rebalance)
        self.max_retries = max_retries
        self.expert_nodes = {entry.node_id for entry in table.assignment.values()}
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.expert_nodes)))

    def __call__(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        for _ in range(self.max_retries + 1):
            chosen = self._chosen(top_k_index)
            try:
                return self._run(layer, hidden_states, top_k_index, top_k_weights, chosen)
            except _NodeFailure as failure:
                logger.warning("%s is down; failing over its experts", failure.node)
                self.table = self.on_failure(failure.node)
                self.expert_nodes = {e.node_id for e in self.table.assignment.values()}
        raise RuntimeError("expert dispatch failed after retries")

    def _chosen(self, top_k_index: Tensor) -> set[str]:
        return {self.table.assignment[e].node_id for e in top_k_index.flatten().tolist()}

    def _run(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
        chosen: set[str],
    ) -> Tensor:
        jobs = [
            self._pool.submit(
                self._send, container, layer, hidden_states, top_k_index, top_k_weights
            )
            for container in chosen
        ]
        total = torch.zeros_like(hidden_states)
        for job in jobs:
            total += job.result()
        return total

    def _send(
        self,
        container: str,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        body = ExpertRequest(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
        ).to_bytes()
        try:
            response = requests.post(f"{address(container)}/{layer}", data=body, timeout=30)
            response.raise_for_status()
        except RequestException as exc:
            raise _NodeFailure(container) from exc
        return ExpertResponse.from_bytes(response.content).out

    def wait_for_experts(self) -> None:
        """Block until every expert node reports it is ready (loaded its experts)."""
        for container in sorted(self.expert_nodes):
            while True:
                try:
                    response = requests.get(f"{address(container)}/health", timeout=2)
                    if response.json().get("ready"):
                        break
                except (RequestException, ValueError):
                    pass
                time.sleep(1)
            logger.info("%s is up", container)

    def close(self) -> None:
        """Release the worker pool."""
        self._pool.shutdown(wait=False)
