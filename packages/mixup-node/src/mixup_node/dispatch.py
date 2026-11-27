"""Central-side dispatcher: fan expert work out to the containers and sum replies.

The expert data path uses a persistent raw-TCP connection per node (see ``transport``);
the control path (``/health``) stays on HTTP. On a connection failure the dispatcher
invokes ``on_failure`` — which the central uses to rebalance and broadcast the new
routing table — then retries against the new owner (Phase 6/7 failover).

Every request is timed and logged; if a ``MetricsCollector`` is attached, each expert
round-trip and layer dispatch is also persisted.
"""

import logging
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from requests.exceptions import RequestException
from torch import Tensor

from mixup_core.constants import EXPERT_PORT, address
from mixup_core.runtime import RoutingTable
from mixup_core.stats import TraceBuffer

from .metrics import MetricsCollector
from .serialization import ExpertRequest, ExpertResponse
from .transport import TCPConnection

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
        stats: TraceBuffer | None = None,
        predictor: object | None = None,
        num_experts: int = 0,
    ) -> None:
        self.name = name
        self.table = table
        self.on_failure = on_failure  # (node) -> new RoutingTable (after rebalance)
        self.max_retries = max_retries
        self.stats = stats  # optional activation-statistics recorder (leader only)
        self.predictor = predictor  # optional next-layer predictor (leader only)
        self.num_experts = num_experts
        self._token_layers: list[list[int]] = []
        self._prev_hidden: Tensor | None = None
        self._prev_experts: list[int] = []
        self.expert_nodes = {entry.node_id for entry in table.assignment.values()}
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.expert_nodes)))
        # A shared Session for the control path (/health polling), and one persistent
        # raw-TCP connection per expert node for the data path.
        self._session = requests.Session()
        self._connections: dict[str, TCPConnection] = {}
        self.metrics: MetricsCollector | None = None
        self._token = 0  # increments each time layer 0 is dispatched (a new token)

    def __call__(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        if layer == 0:
            self._token += 1
            self._prev_hidden = None
            self._prev_experts = []
        # Record activation stats for *decode* tokens only (single-token shape (1, k)).
        # The prefill is batched (shape (prompt_len, k)) and would skew the per-token stats.
        if self.stats is not None and top_k_index.shape[0] == 1:
            if layer == 0:
                self._token_layers = [[] for _ in range(self.stats.num_layers)]
            self._token_layers[layer] = top_k_index[0].tolist()
            if layer == self.stats.num_layers - 1:
                self.stats.record(self._token_layers)
        # Online-train the next-layer predictor (leader only): features = [h_l, one_hot(e_l)],
        # label = the actual top-k at this layer (which is l+1 relative to the previous call).
        if self.predictor is not None and top_k_index.shape[0] == 1:
            cur_experts = top_k_index[0].tolist()
            if self._prev_hidden is not None and self.num_experts > 0:
                from .predictor import _labels, features

                x = features(self._prev_hidden, self._prev_experts, self.num_experts)
                y = _labels(cur_experts, self.num_experts)
                probs = self.predictor.predict(x)
                self.predictor.record_accuracy(probs, cur_experts)
                self.predictor.update(x, y)
            self._prev_hidden = hidden_states[0].detach()
            self._prev_experts = cur_experts
        for _ in range(self.max_retries + 1):
            chosen = self._chosen(top_k_index)
            try:
                return self._run(layer, hidden_states, top_k_index, top_k_weights, chosen)
            except _NodeFailure as failure:
                logger.warning("%s is down; failing over its experts", failure.node)
                self.set_table(self.on_failure(failure.node))
        raise RuntimeError("expert dispatch failed after retries")

    def _chosen(self, top_k_index: Tensor) -> set[str]:
        return {self.table.assignment[e].node_id for e in top_k_index.flatten().tolist()}

    def _prefetch_hint(
        self, container: str, layer: int, hidden_states: Tensor, top_k_index: Tensor
    ) -> list[int]:
        """Predict next-layer experts; return this node's owned ones (top-k)."""
        from .predictor import features

        x = features(hidden_states[0], top_k_index[0].tolist(), self.num_experts)
        probs = self.predictor.predict(x)  # [num_experts] probabilities for layer+1
        owned = [e for e, r in self.table.assignment.items() if r.node_id == container]
        ranked = sorted(owned, key=lambda e: -probs[e].item())
        return ranked[: top_k_index.shape[1]]

    def set_table(self, table: RoutingTable) -> None:
        """Adopt a new routing table (on failover or rejoin), refreshing the expert set."""
        self.table = table
        self.expert_nodes = {entry.node_id for entry in table.assignment.values()}
        # Drop connections to nodes that are no longer experts (rebalanced away).
        for container in list(self._connections):
            if container not in self.expert_nodes:
                self._drop_connection(container)

    def _connection(self, container: str) -> TCPConnection:
        """Return (lazily creating) the persistent data-plane connection to ``container``.

        A short socket timeout bounds how long a dead/unreachable node can stall a
        dispatch before the connection fails and failover kicks in.
        """
        conn = self._connections.get(container)
        if conn is None:
            conn = TCPConnection(container, EXPERT_PORT, timeout=10.0)
            self._connections[container] = conn
        return conn

    def _drop_connection(self, container: str) -> None:
        conn = self._connections.pop(container, None)
        if conn is not None:
            conn.close()

    def _run(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
        chosen: set[str],
    ) -> Tensor:
        t0 = time.perf_counter()
        jobs = [
            self._pool.submit(
                self._send, container, layer, hidden_states, top_k_index, top_k_weights
            )
            for container in chosen
        ]
        # Expert replies are deserialized on CPU (the wire is CPU); accumulate there and
        # move the sum back onto the model's device at the end.
        reduce_start = time.perf_counter()
        total = torch.zeros_like(hidden_states, device="cpu")
        for job in jobs:
            total += job.result()
        if torch.isnan(total).any() or torch.isinf(total).any():
            logger.error("expert reduce produced NaN/inf at token %d layer %d", self._token, layer)
        reduce_time = time.perf_counter() - reduce_start
        dispatch_time = time.perf_counter() - t0

        experts = sorted(chosen)
        logger.info(
            "token %d layer %d: experts=%s dispatch=%.2fms reduce=%.2fms",
            self._token,
            layer,
            experts,
            dispatch_time * 1000,
            reduce_time * 1000,
        )
        if self.metrics is not None:
            self.metrics.record_layer(
                token=self._token,
                layer=layer,
                experts=experts,
                dispatch=dispatch_time,
                reduce_=reduce_time,
            )
        return total.to(hidden_states.device)

    def _send(
        self,
        container: str,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        # Prefetch hint (Phase 6): predict next-layer experts, filtered to this node's
        # owned experts, so it can prefetch layer+1 while computing layer.
        prefetch: list[int] = []
        if self.predictor is not None and top_k_index.shape[0] == 1:
            prefetch = self._prefetch_hint(container, layer, hidden_states, top_k_index)
        serialize_start = time.perf_counter()
        body = ExpertRequest(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            prefetch=torch.tensor(prefetch, dtype=torch.int64),
        ).to_bytes()
        serialize_time = time.perf_counter() - serialize_start

        try:
            http_start = time.perf_counter()
            # Request: [layer][body]; reply: [server_seconds][out], over one persistent
            # connection per node (no HTTP framing per call).
            reply = self._connection(container).request(struct.pack(">I", layer) + body)
            http_time = time.perf_counter() - http_start
        except OSError as exc:
            self._drop_connection(container)
            raise _NodeFailure(container) from exc

        (server_time,) = struct.unpack(">d", reply[:8])

        deserialize_start = time.perf_counter()
        out = ExpertResponse.from_bytes(reply[8:]).out
        deserialize_time = time.perf_counter() - deserialize_start

        if torch.isnan(out).any() or torch.isinf(out).any():
            logger.error(
                "token %d layer %d <- %s returned NaN/inf (server=%.2fms)",
                self._token,
                layer,
                container,
                server_time * 1000,
            )

        logger.debug(
            "token %d layer %d -> %s: rt=%.2fms "
            "(serialize=%.2f http=%.2f server=%.2f deserialize=%.2f)",
            self._token,
            layer,
            container,
            (serialize_time + http_time + deserialize_time) * 1000,
            serialize_time * 1000,
            http_time * 1000,
            server_time * 1000,
            deserialize_time * 1000,
        )
        if self.metrics is not None:
            self.metrics.record_expert(
                token=self._token,
                layer=layer,
                expert=container,
                serialize=serialize_time,
                http=http_time,
                server=server_time,
                deserialize=deserialize_time,
            )
        return out

    def wait_for_experts(self, timeout: float = 30.0) -> None:
        """Wait for each expert to report ready, giving up after ``timeout`` per node.

        A dead node is skipped here: the dispatch fails over to the rebalanced set on
        first contact. A rejoining node gets up to ``timeout`` to finish loading.
        """
        for container in sorted(self.expert_nodes):
            deadline = time.monotonic() + timeout
            ready = False
            while time.monotonic() < deadline:
                try:
                    response = self._session.get(f"{address(container)}/health", timeout=2)
                    if response.json().get("ready"):
                        ready = True
                        break
                except (RequestException, ValueError):
                    pass
                time.sleep(1)
            if ready:
                logger.info("%s is up", container)
            else:
                logger.warning(
                    "expert %s not ready after %.0fs; will fail over on dispatch",
                    container,
                    timeout,
                )

    def close(self) -> None:
        """Release the worker pool, control session, and data-plane connections."""
        self._pool.shutdown(wait=False)
        self._session.close()
        for container in list(self._connections):
            self._drop_connection(container)
