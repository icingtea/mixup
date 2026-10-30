import docker
from pathlib import Path
import models
from enum import Enum
import json

config: models.Config = models.Config.model_validate_json(Path("src", "config.json").read_text())

NETWORK_NAME = "cluster"
IMAGE_NAME = "agent-test"
IMAGE_TAG = "centralized"

N_CONTAINERS = len(config.experts)
INTERNAL_PORT = 8000
EXTERNAL_PORT_BASE = 9000

NETWORK_ADDRESSES = {config.central: EXTERNAL_PORT_BASE}
CONTAINERS = {}

for offset, (container_name, experts) in enumerate(config.experts.items()):
    port = EXTERNAL_PORT_BASE + offset + 1
    NETWORK_ADDRESSES[container_name] = port

class ContainerType(Enum):
    CENTRAL = "central"
    EXPERT = "expert"

client = docker.from_env()

image, logs = client.images.build(
    path=".",
    dockerfile=str(Path("src", "load_network", "Dockerfile")),
    tag=f"{IMAGE_NAME}:{IMAGE_TAG}"
)

for log in logs:
    if "stream" in log:
        print(log["stream"], end="")

print(f"BUILT IMAGE {IMAGE_NAME}:{IMAGE_TAG}")

try:
    network = client.networks.get(NETWORK_NAME)
    print("FOUND NETWORK:", NETWORK_NAME)
except:
    network = client.networks.create(
        name=NETWORK_NAME,
        driver="bridge"
    )

def start_container(port, container_name, type: ContainerType, layers=None):
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
            "type":type.value,
            "layers":",".join(map(str, layers or [])),
        }
    )

    print("STARTED", container_name)
    CONTAINERS[container_name] = container
    NETWORK_ADDRESSES[container_name] = port

    return True


start_container(EXTERNAL_PORT_BASE, config.central, ContainerType.CENTRAL)

for offset, (container_name, experts) in enumerate(config.experts.items()):
    start_container(NETWORK_ADDRESSES[container_name], container_name, ContainerType.EXPERT, layers=experts)
