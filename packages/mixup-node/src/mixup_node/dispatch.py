"""Central-side HTTP client: fan expert work out to the containers and sum replies."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from requests.exceptions import RequestException
from torch import Tensor

from mixup_core.constants import address
from mixup_core.topology import expert_containers
from mixup_core.types import Config, ExpertRoute

from .serialization import ExpertRequest, ExpertResponse

logger = logging.getLogger(__name__)


class Dispatcher:
    """Runs each token's chosen experts on the containers that hold them, in parallel."""

    def __init__(self, config: Config, lut: dict[int, ExpertRoute]) -> None:
        self.lut = lut
        self.containers = expert_containers(config)
        self._pool = ThreadPoolExecutor(max_workers=len(self.containers))

    def __call__(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        chosen = {self.lut[e][0] for e in top_k_index.flatten().tolist()}
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
        response = requests.post(f"{address(container)}/{layer}", data=body)
        response.raise_for_status()
        return ExpertResponse.from_bytes(response.content).out

    def wait_for_experts(self) -> None:
        """Block until every expert container answers."""
        for container in self.containers:
            while True:
                try:
                    requests.get(address(container), timeout=2)
                    break
                except RequestException:
                    time.sleep(1)
            logger.info("%s is up", container)

    def close(self) -> None:
        """Release the worker pool."""
        self._pool.shutdown(wait=False)
