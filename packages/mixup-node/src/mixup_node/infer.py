"""Node entrypoint: run as the central or an expert container.

Run inside every container. The runtime environment (container name, cluster layout,
prompt, token budget) is read from the single MIXUP_NODE_ENV env var, deserialized
into a typed NodeEnv at bootup.
"""

import logging

import torch
from transformers import AutoTokenizer, GraniteMoeConfig

from mixup_core.constants import INTERNAL_PORT, MODEL_DIR, MODEL_FILE
from mixup_core.env import NodeEnv, load_node_env
from mixup_core.topology import build_lut, validate_partition
from mixup_core.types import ExpertRoute, NodeRole

from .dispatch import Dispatcher
from .model import ExpertGroup, load_central
from .server import serve

logger = logging.getLogger(__name__)


def _model_config() -> GraniteMoeConfig:
    return GraniteMoeConfig.from_pretrained(MODEL_DIR)


def run_central(
    env: NodeEnv,
    model_config: GraniteMoeConfig,
    lut: dict[int, ExpertRoute],
) -> None:
    """Load the central model, wait for the experts, and generate one completion."""
    dispatcher = Dispatcher(env.config, lut)
    model = load_central(model_config, MODEL_FILE, dispatcher)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dispatcher.wait_for_experts()

    ids = tokenizer(env.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=env.max_new_tokens, do_sample=False)
    logger.info(tokenizer.decode(out[0][ids.shape[1] :]))


def run_expert(
    env: NodeEnv,
    model_config: GraniteMoeConfig,
    lut: dict[int, ExpertRoute],
) -> None:
    """Load this container's experts and serve them over HTTP."""
    group = ExpertGroup(env.container_name, model_config, MODEL_FILE, lut)
    logger.info(
        "%s: experts %d-%d ready on port %d",
        env.container_name,
        group.experts[0],
        group.experts[-1],
        INTERNAL_PORT,
    )
    serve(group, INTERNAL_PORT)


def start() -> None:
    """Deserialize MIXUP_NODE_ENV and start as central or expert."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env = load_node_env()
    model_config = _model_config()
    lut = build_lut(env.config)
    validate_partition(env.config, model_config.num_local_experts or 0)

    if env.config.initial_role(env.container_name) is NodeRole.LEADER:
        run_central(env, model_config, lut)
    else:
        run_expert(env, model_config, lut)


if __name__ == "__main__":
    start()
