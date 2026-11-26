"""Node entrypoint: run as the central/leader or an expert node.

Run inside every container. The runtime environment (container name, startup layout,
prompt, token budget) is read from the single MIXUP_NODE_ENV env var, deserialized
into a typed NodeEnv at bootup.
"""

import logging

import requests
import torch
from requests.exceptions import RequestException
from transformers import AutoTokenizer, GraniteMoeConfig

from mixup_core.constants import INTERNAL_PORT, MODEL_DIR, MODEL_FILE, address
from mixup_core.env import NodeEnv, load_node_env
from mixup_core.runtime import ClusterView, expert_moves
from mixup_core.scheduler import handle_node_failure
from mixup_core.topology import cluster_view_from_config, validate_partition
from mixup_core.types import MemberStatus, NodeRole

from .dispatch import Dispatcher
from .model import ExpertGroup, load_central
from .server import serve

logger = logging.getLogger(__name__)


def _model_config() -> GraniteMoeConfig:
    return GraniteMoeConfig.from_pretrained(MODEL_DIR)


def broadcast_routing(view: ClusterView, by_node: str) -> None:
    """Push the current routing table to every alive expert node (so they load new experts)."""
    table = view.routing
    if table is None:
        return
    payload = table.model_dump()
    for member in view.members.values():
        if member.role is NodeRole.EXPERT and member.status is MemberStatus.ALIVE:
            try:
                requests.post(f"{address(member.node_id)}/routing", json=payload, timeout=60)
                logger.info("broadcast: sent routing epoch %d to %s", table.epoch, member.node_id)
            except RequestException:
                logger.warning("broadcast: could not reach %s during rebalance", member.node_id)


def run_central(env: NodeEnv, model_config: GraniteMoeConfig, view: ClusterView) -> None:
    """Load the central model, wait for the experts, and generate one completion."""

    def on_failure(node: str):
        nonlocal view
        old = view.routing
        view = handle_node_failure(view, node, by_node=env.container_name)
        new = view.routing
        assert old is not None and new is not None

        moves = expert_moves(old, new)
        by_owner: dict[str, list[int]] = {}
        for expert, _old_owner, owner in moves:
            by_owner.setdefault(owner, []).append(expert)

        logger.warning(
            "FAILOVER: %s is dead; reassigned %d experts (routing epoch %d)",
            node,
            len(moves),
            new.epoch,
        )
        for owner in sorted(by_owner):
            logger.info("  -> %s now holds experts %s", owner, sorted(by_owner[owner]))

        broadcast_routing(view, env.container_name)
        return new

    assert view.routing is not None
    dispatcher = Dispatcher(env.container_name, view.routing, on_failure)
    model = load_central(model_config, MODEL_FILE, dispatcher)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dispatcher.wait_for_experts()

    ids = tokenizer(env.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=env.max_new_tokens, do_sample=False)
    logger.info("generated: %s", tokenizer.decode(out[0][ids.shape[1] :]))


def run_expert(env: NodeEnv, model_config: GraniteMoeConfig, view: ClusterView) -> None:
    """Load this node's experts and serve them (with failover reload) over HTTP."""
    assert view.routing is not None
    group = ExpertGroup(env.container_name, model_config, MODEL_FILE, view.routing)
    logger.info(
        "%s: %d experts ready on port %d",
        env.container_name,
        len(group.experts),
        INTERNAL_PORT,
    )
    serve(group, INTERNAL_PORT)


def start() -> None:
    """Deserialize MIXUP_NODE_ENV and start as the leader or an expert node."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    env = load_node_env()
    model_config = _model_config()
    validate_partition(env.config, model_config.num_local_experts or 0)
    view = cluster_view_from_config(env.config)

    role = env.config.initial_role(env.container_name)
    logger.info("%s boot: role=%s, epoch=0", env.container_name, role.value)
    assert view.routing is not None
    if role is NodeRole.LEADER:
        owners = {r.node_id for r in view.routing.assignment.values()}
        logger.info(
            "%s boot: %d experts across %d expert nodes",
            env.container_name,
            len(view.routing.assignment),
            len(owners),
        )
    else:
        experts = sorted(
            e for e, r in view.routing.assignment.items() if r.node_id == env.container_name
        )
        logger.info("%s boot: experts=%s", env.container_name, experts)

    if role is NodeRole.LEADER:
        run_central(env, model_config, view)
    else:
        run_expert(env, model_config, view)


if __name__ == "__main__":
    start()
