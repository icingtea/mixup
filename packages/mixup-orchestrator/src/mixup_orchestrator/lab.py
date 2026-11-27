"""``mixup-lab``: stand up a mixup cluster (emulated on this machine, or across devices).

The config decides the mode: a config with an empty ``startup.endpoints`` map runs the
single-host emulated lab (every node on the local Docker daemon, sharing the cluster
network, with bind-mounted models); a config with explicit endpoints deploys one container
per device (each publishing its control + expert ports, using named volumes, reachable over
the configured URLs). The pure run plan lives in ``deploy.py``; ``targets.py`` executes it.
"""

import argparse
import logging
from pathlib import Path

import docker

from mixup_core.constants import NETWORK_NAME
from mixup_core.lab import LabConfig
from mixup_core.logging_config import configure_logging

from . import targets
from .network import get_or_create_network

logger = logging.getLogger(__name__)


def _load_lab_config(path: str | Path) -> LabConfig:
    return LabConfig.model_validate_json(Path(path).read_text())


def _clients(specs: dict) -> dict[str, docker.DockerClient]:
    """One Docker client per distinct ``docker_host``, cached by host."""
    clients: dict[str, docker.DockerClient] = {}
    for spec in specs.values():
        clients.setdefault(spec.docker_host, targets.get_client(spec.docker_host))
    return clients


def _cmd_up(lab: LabConfig, prompt: str, max_new_tokens: int) -> int:
    from .deploy import plan_run_specs

    specs = plan_run_specs(lab, prompt, max_new_tokens)
    # The emulated lab needs the cluster network so container names resolve.
    if any(spec.network == NETWORK_NAME for spec in specs.values()):
        get_or_create_network(targets.get_client("local"))

    clients = _clients(specs)
    for spec in specs.values():
        client = clients[spec.docker_host]
        targets.ensure_image(client, spec)
        targets.ensure_model(client, spec)
        targets.ensure_ledger(client, spec)
        targets.start(client, spec)
    return 0


def _cmd_down(lab: LabConfig, prompt: str, max_new_tokens: int) -> int:
    from .deploy import plan_run_specs

    specs = plan_run_specs(lab, prompt, max_new_tokens)
    clients = _clients(specs)
    for spec in specs.values():
        targets.remove(clients[spec.docker_host], spec)
    return 0


def _cmd_status(lab: LabConfig, prompt: str, max_new_tokens: int) -> int:
    from .deploy import plan_run_specs

    specs = plan_run_specs(lab, prompt, max_new_tokens)
    clients = _clients(specs)
    for spec in specs.values():
        print(f"{spec.container_name:20s} {targets.status(clients[spec.docker_host], spec)}")
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


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Stand up the mixup lab / cluster.")
    parser.add_argument(
        "command_or_config",
        nargs="?",
        default=None,
        help="one of: up, down, status, render, doctor (else the config path for `up`).",
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

    commands = {"up", "down", "status", "render", "doctor"}
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
    if command == "down":
        return _cmd_down(lab, args.prompt, args.max_new_tokens)
    if command == "status":
        return _cmd_status(lab, args.prompt, args.max_new_tokens)
    return _cmd_up(lab, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
