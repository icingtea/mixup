"""Node entrypoint: run as the central or an expert container, depending on CONTAINER_NAME.

Run inside every container. The central container loads everything except the experts
and fans expert work out; each expert container loads only its own experts and serves
them over HTTP.
"""

import logging
import os

import torch
from transformers import AutoTokenizer, GraniteMoeConfig

from mixup_core.config import (
    MAX_NEW_TOKENS_ENV,
    PROMPT_ENV,
    container_name_from_env,
    load_config_from_env,
)
from mixup_core.constants import INTERNAL_PORT, MODEL_DIR, MODEL_FILE
from mixup_core.topology import build_lut, validate_partition
from mixup_core.types import Config, ContainerType, ExpertRoute

from .dispatch import Dispatcher
from .model import ExpertGroup, load_central
from .server import serve

logger = logging.getLogger(__name__)


def _model_config() -> GraniteMoeConfig:
    return GraniteMoeConfig.from_pretrained(MODEL_DIR)


def run_central(
    config: Config,
    model_config: GraniteMoeConfig,
    lut: dict[int, ExpertRoute],
) -> None:
    """Load the central model, wait for the experts, and generate one completion."""
    dispatcher = Dispatcher(config, lut)
    model = load_central(model_config, MODEL_FILE, dispatcher)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dispatcher.wait_for_experts()

    prompt = os.getenv(PROMPT_ENV, "What is mixture of experts?")
    max_new_tokens = int(os.getenv(MAX_NEW_TOKENS_ENV, "50"))
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    logger.info(tokenizer.decode(out[0][ids.shape[1] :]))


def run_expert(
    name: str,
    model_config: GraniteMoeConfig,
    lut: dict[int, ExpertRoute],
) -> None:
    """Load this container's experts and serve them over HTTP."""
    group = ExpertGroup(name, model_config, MODEL_FILE, lut)
    logger.info(
        "%s: experts %d-%d ready on port %d",
        name,
        group.experts[0],
        group.experts[-1],
        INTERNAL_PORT,
    )
    serve(group, INTERNAL_PORT)


def start() -> None:
    """Start as central or expert, depending on CONTAINER_NAME."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    name = container_name_from_env()
    config = load_config_from_env()
    model_config = _model_config()
    lut = build_lut(config)
    validate_partition(config, model_config.num_local_experts or 0)

    if config.role(name) is ContainerType.CENTRAL:
        run_central(config, model_config, lut)
    else:
        run_expert(name, model_config, lut)


if __name__ == "__main__":
    start()
