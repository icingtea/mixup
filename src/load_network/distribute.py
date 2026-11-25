"""Builds the Docker image and starts one container per entry in src/config.json.

Run from the project root. Every container gets the same image; CONTAINER_NAME and
CONFIG tell infer.py which part of the model it holds. The model folder is mounted
read-only into each container instead of being copied into the image.
"""

import docker
from docker.models.containers import Container
from docker.models.images import Image
from docker.models.networks import Network
from pathlib import Path
import models
from enum import Enum
import json

config: models.Config = models.Config.model_validate_json(Path("src", "config.json").read_text())

NETWORK_NAME = "cluster"
IMAGE_NAME = "agent-test"
IMAGE_TAG = "centralized"

N_CONTAINERS = len(config.experts)
INTERNAL_PORT = 8000  # the port infer.py listens on inside every container
EXTERNAL_PORT_BASE = 9000  # central gets 9000 on the host, expert containers 9001, 9002, ...

NETWORK_ADDRESSES: dict[str, int] = {config.central: EXTERNAL_PORT_BASE}
"""Container name -> its port on the host machine."""

CONTAINERS: dict[str, Container] = {}
"""Containers started by this run."""

for offset, (container_name, experts) in enumerate(config.experts.items()):
    port = EXTERNAL_PORT_BASE + offset + 1
    NETWORK_ADDRESSES[container_name] = port


class ContainerType(Enum):
    """What a container runs: everything except the experts, or some experts."""

    CENTRAL = "central"
    EXPERT = "expert"


client = docker.from_env()

image: Image
image, logs = client.images.build(
    path=".",
    dockerfile=str(Path("src", "load_network", "Dockerfile")),
    tag=f"{IMAGE_NAME}:{IMAGE_TAG}",
    rm=True,
    forcerm=True
)

for log in logs:
    if "stream" in log:
        print(log["stream"], end="")

print(f"BUILT IMAGE {IMAGE_NAME}:{IMAGE_TAG}")

network: Network
try:
    network = client.networks.get(NETWORK_NAME)
    print("FOUND NETWORK:", NETWORK_NAME)
except:
    network = client.networks.create(
        name=NETWORK_NAME,
        driver="bridge"
    )


def start_container(
    port: int,
    container_name: str,
    type: ContainerType,
    layers: list[int] | None = None,
) -> bool:
    """Start `container_name` on the cluster network, reachable from the host on `port`.

    If a container with that name already exists, it is started again (or left running)
    instead of being recreated, so changes to the image only apply to new containers.
    """
    try:
        container = client.containers.get(container_name)
        if container.status == "running":
            print(container_name, "RUNNING.")
        else:
            container.start()
            print(container_name, "STARTED.")

        return True
    except:
        pass


    container = client.containers.run(
        image=image,
        name=container_name,
        detach=True,
        network=NETWORK_NAME,
        ports={
            f"{INTERNAL_PORT}/tcp": port
        },
        restart_policy={
            "Name": "unless-stopped"
        },
        environment={
            "CONTAINER_NAME": container_name,
            "CONFIG": config.model_dump_json(),
        },
        volumes={
            str(Path("models").resolve()): {"bind": "/usr/local/app/models", "mode": "ro"}
        }
    )

    print("STARTED", container_name)
    CONTAINERS[container_name] = container
    NETWORK_ADDRESSES[container_name] = port

    return True


start_container(EXTERNAL_PORT_BASE, config.central, ContainerType.CENTRAL)

for offset, (container_name, experts) in enumerate(config.experts.items()):
    start_container(NETWORK_ADDRESSES[container_name], container_name, ContainerType.EXPERT, layers=experts)
