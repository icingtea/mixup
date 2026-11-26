"""Per-request metrics collection and persistence.

Every expert round-trip, every layer dispatch, and every generation is recorded as one
JSON line (JSONL) in ``ledger/metrics_<node>.jsonl`` — appended, not overwritten, so runs
accumulate and can be analyzed with pandas/jq. This module is torch-free.
"""

import json
import threading
import uuid
from pathlib import Path

from mixup_core.constants import LEDGER_DIR


class MetricsCollector:
    """Accumulates and persists metrics for one generation (one ``/generate`` call).

    Records three kinds of JSONL lines, all keyed by a shared ``generation_id``:

    - ``type=expert`` — one per expert HTTP request (serialize / http / server / deserialize).
    - ``type=layer`` — one per layer dispatch (which experts, dispatch + reduce time).
    - ``type=generation`` — one summary per generation (tokens, total/dense/dispatch, tok/s).
    """

    def __init__(
        self,
        node_name: str,
        prompt: str,
        max_new_tokens: int,
        path: Path | None = None,
    ) -> None:
        self.generation_id = uuid.uuid4().hex[:12]
        self.node_name = node_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.dispatch_seconds = 0.0
        self.expert_requests = 0
        self._path: Path = path or LEDGER_DIR / f"metrics_{node_name}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a")
        self._lock = threading.Lock()

    def _write(self, record: dict) -> None:
        line = json.dumps(record) + "\n"
        with self._lock:
            self._file.write(line)

    def record_expert(
        self,
        *,
        token: int,
        layer: int,
        expert: str,
        serialize: float,
        http: float,
        server: float,
        deserialize: float,
    ) -> None:
        """One expert round-trip. ``network`` = http minus the server's own handling time."""
        self.expert_requests += 1
        self._write(
            {
                "type": "expert",
                "generation_id": self.generation_id,
                "node": self.node_name,
                "token": token,
                "layer": layer,
                "expert": expert,
                "serialize_seconds": round(serialize, 6),
                "http_seconds": round(http, 6),
                "server_seconds": round(server, 6),
                "deserialize_seconds": round(deserialize, 6),
                "network_seconds": round(http - server, 6),
                "round_trip_seconds": round(serialize + http + deserialize, 6),
            }
        )

    def record_layer(
        self,
        *,
        token: int,
        layer: int,
        experts: list[str],
        dispatch: float,
        reduce_: float,
    ) -> None:
        """One layer's dispatch: which experts were contacted, and dispatch + reduce time."""
        self.dispatch_seconds += dispatch
        self._write(
            {
                "type": "layer",
                "generation_id": self.generation_id,
                "node": self.node_name,
                "token": token,
                "layer": layer,
                "experts": experts,
                "num_experts": len(experts),
                "dispatch_seconds": round(dispatch, 6),
                "reduce_seconds": round(reduce_, 6),
            }
        )

    def record_generation(
        self,
        *,
        tokens: int,
        total: float,
        tok_per_sec: float,
    ) -> None:
        """Summary line. ``dense_seconds`` is total minus the distributed dispatch time."""
        dense = max(0.0, total - self.dispatch_seconds)
        self._write(
            {
                "type": "generation",
                "generation_id": self.generation_id,
                "node": self.node_name,
                "prompt": self.prompt,
                "max_new_tokens": self.max_new_tokens,
                "tokens": tokens,
                "total_seconds": round(total, 6),
                "dense_seconds": round(dense, 6),
                "dispatch_seconds": round(self.dispatch_seconds, 6),
                "expert_requests": self.expert_requests,
                "tokens_per_second": round(tok_per_sec, 3),
            }
        )
        self._file.flush()

    def close(self) -> None:
        """Close the backing file (idempotent)."""
        if not self._file.closed:
            self._file.close()
