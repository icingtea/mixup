"""``mixup-lab``: stand up an emulated heterogeneous cluster on this machine.

Unlike ``mixup-orchestrator`` (which gives every container the GPU), the lab starts each
node with its own ``LabNodeSpec``: GPU nodes get the NVIDIA CDI device, CPU nodes run with
``CUDA_VISIBLE_DEVICES=""`` and no GPU device, and both can be capped with ``--cpus`` /
``--memory``. The node's ``DeviceProfile`` (for placement) is injected via ``NodeEnv``.
"""

import argparse
import logging
from pathlib import Path

import docker
from docker.errors import NotFound
from docker.types import DeviceRequest

from mixup_core.constants import (
    EXTERNAL_PORT_BASE,
    IMAGE_NAME,
    IMAGE_TAG,
    INTERNAL_PORT,
    NETWORK_NAME,
)
from mixup_core.devices import ExecutionType
from mixup_core.env import NODE_ENV_VAR, NodeEnv, dump_node_env
from mixup_core.lab import LabConfig
from mixup_core.logging_config import configure_logging

from .network import get_or_create_network

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CONTAINER_MODELS_DIR = "/usr/local/app/models"
LEDGER_DIR = Path("var/ledger")
CONTAINER_LEDGER_DIR = "/usr/local/app/ledger"


def _load_lab_config(path: str | Path) -> LabConfig:
    return LabConfig.model_validate_json(Path(path).read_text())


def _start_node(
    client: docker.DockerClient,
    image,
    name: str,
    port: int,
    lab: LabConfig,
    prompt: str,
    max_new_tokens: int,
) -> None:
    try:
        container = client.containers.get(name)
        if container.status == "running":
            logger.info("%s RUNNING.", name)
        else:
            container.start()
            logger.info("%s STARTED.", name)
        return
    except NotFound:
        pass

    spec = lab.spec_for(name)
    is_gpu = spec.execution is ExecutionType.GPU
    device_requests = (
        [DeviceRequest(driver="cdi", device_ids=["nvidia.com/gpu=all"])] if is_gpu else None
    )

    node_env = NodeEnv(
        container_name=name,
        config=lab.startup,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        profiles=lab.profiles,
        profile_mode=lab.profile_mode,
    )
    environment = {NODE_ENV_VAR: dump_node_env(node_env)}
    if not is_gpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.chmod(0o777)

    client.containers.run(
        image=image,
        name=name,
        detach=True,
        network=NETWORK_NAME,
        ports={f"{INTERNAL_PORT}/tcp": port},
        restart_policy={"Name": "unless-stopped"},
        environment=environment,
        volumes={
            str(MODELS_DIR.resolve()): {"bind": CONTAINER_MODELS_DIR, "mode": "ro"},
            str(LEDGER_DIR.resolve()): {"bind": CONTAINER_LEDGER_DIR, "mode": "rw"},
        },
        device_requests=device_requests,
        nano_cpus=int(spec.cpus * 1e9) if spec.cpus else None,
        mem_limit=f"{spec.memory_mb}m" if spec.memory_mb else None,
    )
    logger.info(
        "STARTED %s (%s, cpus=%s mem=%s)",
        name,
        spec.execution.value,
        spec.cpus,
        spec.memory_mb,
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Stand up the heterogeneous lab.")
    parser.add_argument(
        "command_or_config",
        nargs="?",
        default=None,
        help="one of: render, doctor (else the config path for `up`).",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="path to a lab config (lab_config.json / devices.json).",
    )
    parser.add_argument("--prompt", default="What is mixture of experts?")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args(argv)

    commands = {"up", "render", "doctor"}
    if args.command_or_config in commands:
        command = args.command_or_config
        config_path = args.config or "lab_config.json"
    else:
        command = "up"
        config_path = args.command_or_config or "lab_config.json"

    lab = _load_lab_config(config_path)

    if command == "render":
        return _cmd_render(lab, args.prompt, args.max_new_tokens)
    if command == "doctor":
        return _cmd_doctor(lab, args.prompt, args.max_new_tokens)

    client = docker.from_env()
    image = client.images.get(f"{IMAGE_NAME}:{IMAGE_TAG}")
    get_or_create_network(client)

    leader = lab.startup.initial_leader
    _start_node(client, image, leader, EXTERNAL_PORT_BASE, lab, args.prompt, args.max_new_tokens)
    others = [n for n in lab.startup.nodes if n != leader]
    for offset, name in enumerate(others):
        _start_node(
            client,
            image,
            name,
            EXTERNAL_PORT_BASE + offset + 1,
            lab,
            args.prompt,
            args.max_new_tokens,
        )
    return 0


def _cmd_render(lab: LabConfig, prompt: str, max_new_tokens: int) -> int:
    """Print an equivalent ``docker run`` command per node (pure; no Docker needed)."""
    from .deploy import plan_run_specs, render_command

    specs = plan_run_specs(lab, prompt, max_new_tokens)
    for node, spec in specs.items():
        print(f"# {node} ({spec.docker_host})")
        print(render_command(spec))
        print()
    return 0


def _cmd_doctor(lab: LabConfig, prompt: str, max_new_tokens: int) -> int:
    """Validate the config by planning every node, and print a summary (docker-free)."""
    from .deploy import plan_run_specs

    specs = plan_run_specs(lab, prompt, max_new_tokens)
    print(f"cluster={lab.cluster_name!r} image={lab.image!r}")
    print(f"leader={lab.startup.initial_leader} nodes={len(lab.startup.nodes)}")
    for node, spec in specs.items():
        ports = ", ".join(
            f"{p.container_port}->{p.bind_ip or '0.0.0.0'}:{p.host_port}" for p in spec.ports
        )
        print(
            f"  {node:16s} host={spec.docker_host:24s} gpu={spec.gpu.value:6s} "
            f"net={spec.network} ports=[{ports}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
