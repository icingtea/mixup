"""Host-side orchestrator: build the image and start one container per config entry.

Run from the project root. Every container gets the same image; CONTAINER_NAME and
CONFIG tell each node which part of the model it holds. The models/ directory is
mounted read-only into each container instead of being copied into the image.
"""

import logging
import os
import sys
from pathlib import Path

import docker
from docker.errors import BuildError, NotFound
from docker.models.images import Image
from docker.models.networks import Network

from mixup_core.config import (
    CONFIG_ENV,
    CONTAINER_NAME_ENV,
    MAX_NEW_TOKENS_ENV,
    PROMPT_ENV,
    load_config_file,
)
from mixup_core.constants import (
    EXTERNAL_PORT_BASE,
    IMAGE_NAME,
    IMAGE_TAG,
    INTERNAL_PORT,
    MODEL_FILE,
    NETWORK_NAME,
)
from mixup_core.types import Config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.json")
MODELS_DIR = Path("models")
# Matches the Dockerfile WORKDIR (/usr/local/app) + MODEL_ROOT ("models").
CONTAINER_MODELS_DIR = "/usr/local/app/models"


def build_image(client: docker.DockerClient) -> Image:
    """Build the node image, streaming its output in real time."""
    logger.info(
        "BUILDING IMAGE %s:%s (first build downloads torch, takes a while)...",
        IMAGE_NAME,
        IMAGE_TAG,
    )
    # client.api.build() streams live; client.images.build() buffers until the end.
    stream = client.api.build(
        path=".",
        dockerfile="Dockerfile",
        tag=f"{IMAGE_NAME}:{IMAGE_TAG}",
        rm=True,
        forcerm=True,
        decode=True,
    )
    for chunk in stream:
        if "stream" in chunk:
            logger.info("%s", chunk["stream"].rstrip("\n"))
        elif "status" in chunk:
            detail = chunk.get("progress") or chunk.get("id") or ""
            logger.info("%s %s", chunk["status"], detail)
        elif "error" in chunk:
            raise BuildError(chunk["error"], None)
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
    config: Config,
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

    environment = {
        CONTAINER_NAME_ENV: name,
        CONFIG_ENV: config.model_dump_json(),
    }
    prompt = os.getenv(PROMPT_ENV)
    if prompt:
        environment[PROMPT_ENV] = prompt
    max_new_tokens = os.getenv(MAX_NEW_TOKENS_ENV)
    if max_new_tokens:
        environment[MAX_NEW_TOKENS_ENV] = max_new_tokens

    client.containers.run(
        image=image,
        name=name,
        detach=True,
        network=NETWORK_NAME,
        ports={f"{INTERNAL_PORT}/tcp": port},
        restart_policy={"Name": "unless-stopped"},
        environment=environment,
        volumes={str(MODELS_DIR.resolve()): {"bind": CONTAINER_MODELS_DIR, "mode": "ro"}},
    )
    logger.info("STARTED %s", name)


def main(argv: list[str] | None = None) -> int:
    """Build the image and start every container in the config."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv if argv is not None else sys.argv[1:]
    config_path = Path(args[0]) if args else DEFAULT_CONFIG_PATH
    config = load_config_file(config_path)

    if not MODEL_FILE.exists():
        logger.error("model not found at %s", MODEL_FILE)
        logger.error("download it first with: mixup-fetch-model")
        return 1

    client = docker.from_env()
    image = build_image(client)
    get_or_create_network(client)

    start_container(client, image, config.central, EXTERNAL_PORT_BASE, config)
    for offset, container_name in enumerate(config.experts):
        start_container(client, image, container_name, EXTERNAL_PORT_BASE + offset + 1, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
