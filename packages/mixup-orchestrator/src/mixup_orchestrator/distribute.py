"""Host-side orchestrator: build the image and start one container per config entry.

Run from the project root. Every container gets the same image and one env var,
MIXUP_NODE_ENV (a serialized NodeEnv), which tells the node its identity, the cluster
layout, and the generation settings. The models/ directory is mounted read-only into
each container instead of being copied into the image.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import docker
from docker.errors import BuildError, NotFound
from docker.models.images import Image
from docker.models.networks import Network
from docker.types import DeviceRequest

from mixup_core.constants import (
    EXTERNAL_PORT_BASE,
    IMAGE_NAME,
    IMAGE_TAG,
    INTERNAL_PORT,
    MODEL_FILE,
    NETWORK_NAME,
)
from mixup_core.env import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PROMPT,
    NODE_ENV_VAR,
    NodeEnv,
    dump_node_env,
)
from mixup_core.startup_config import load_startup_config
from mixup_core.types import StartupConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("startup_config.json")
MODELS_DIR = Path("models")
# Matches the Dockerfile WORKDIR (/usr/local/app) + MODEL_ROOT ("models").
CONTAINER_MODELS_DIR = "/usr/local/app/models"
# Read-write ledger volume for the KV/checkpoint resume state.
LEDGER_DIR = Path("var/ledger")
CONTAINER_LEDGER_DIR = "/usr/local/app/ledger"


def build_image(client: docker.DockerClient) -> Image:
    """Build the node image via `docker build` (BuildKit), streaming output live.

    The SDK's legacy /build endpoint does not run BuildKit, so RUN --mount cache
    mounts (and fast code-only rebuilds) don't work there. We shell out to the CLI,
    which uses the BuildKit builder and streams build output directly.

    ``--network=host`` is required: Docker's default bridge DNS cannot resolve the
    CPU-torch wheel host (``download-r2.pytorch.org``), so the build RUN steps use the
    host's DNS (which can).
    """
    logger.info(
        "BUILDING IMAGE %s:%s (first build downloads torch, takes a while)...",
        IMAGE_NAME,
        IMAGE_TAG,
    )
    proc = subprocess.run(
        ["docker", "build", "--network=host", "--tag", f"{IMAGE_NAME}:{IMAGE_TAG}", "."]
    )
    if proc.returncode != 0:
        raise BuildError(f"docker build failed with exit status {proc.returncode}", None)
    logger.info("BUILT IMAGE %s:%s", IMAGE_NAME, IMAGE_TAG)
    return client.images.get(f"{IMAGE_NAME}:{IMAGE_TAG}")


def get_or_create_network(client: docker.DockerClient) -> Network:
    """Return the cluster network, creating it if it does not exist."""
    try:
        network = client.networks.get(NETWORK_NAME)
        logger.info("FOUND NETWORK: %s", NETWORK_NAME)
    except NotFound:
        network = client.networks.create(name=NETWORK_NAME, driver="bridge")
        logger.info("CREATED NETWORK: %s", NETWORK_NAME)
    return network


def start_container(
    client: docker.DockerClient,
    image: Image,
    name: str,
    port: int,
    config: StartupConfig,
    prompt: str,
    max_new_tokens: int,
    device_requests: list[DeviceRequest] | None = None,
) -> None:
    """Start `name` on the cluster network, reachable from the host on `port`."""
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

    node_env = NodeEnv(
        container_name=name,
        config=config,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    # The container runs as a non-root `app` user; make the ledger dir writable by any
    # UID so the bind mount (owned by the host user) is usable for checkpoint writes.
    LEDGER_DIR.chmod(0o777)
    client.containers.run(
        image=image,
        name=name,
        detach=True,
        network=NETWORK_NAME,
        ports={f"{INTERNAL_PORT}/tcp": port},
        restart_policy={"Name": "unless-stopped"},
        environment={NODE_ENV_VAR: dump_node_env(node_env)},
        volumes={
            str(MODELS_DIR.resolve()): {"bind": CONTAINER_MODELS_DIR, "mode": "ro"},
            str(LEDGER_DIR.resolve()): {"bind": CONTAINER_LEDGER_DIR, "mode": "rw"},
        },
        device_requests=device_requests,
    )
    logger.info("STARTED %s", name)


def main(argv: list[str] | None = None) -> int:
    """Build the image and start every container in the config."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build the image and start the cluster.")
    parser.add_argument(
        "config", nargs="?", default=str(DEFAULT_CONFIG_PATH), help="path to config.json"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt for the central node")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="tokens for the central node to generate",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="run CPU-only (don't pass through the GPU)",
    )
    args = parser.parse_args(argv)

    config = load_startup_config(args.config)

    if not MODEL_FILE.exists():
        logger.error("model not found at %s", MODEL_FILE)
        logger.error("download it first with: mixup-fetch-model")
        return 1

    # Use CDI (nvidia.com/gpu=all) rather than --gpus all: --gpus does vendor
    # auto-detection, which breaks on hosts with a mixed AMD iGPU + NVIDIA dGPU
    # ("AMD CDI spec not found"). CDI names the NVIDIA device explicitly.
    device_requests = (
        None
        if args.no_gpu
        else [DeviceRequest(driver="cdi", device_ids=["nvidia.com/gpu=all"])]
    )

    client = docker.from_env()
    image = build_image(client)
    get_or_create_network(client)

    start_container(
        client,
        image,
        config.initial_leader,
        EXTERNAL_PORT_BASE,
        config,
        args.prompt,
        args.max_new_tokens,
        device_requests,
    )
    for offset, container_name in enumerate(config.experts):
        start_container(
            client,
            image,
            container_name,
            EXTERNAL_PORT_BASE + offset + 1,
            config,
            args.prompt,
            args.max_new_tokens,
            device_requests,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
