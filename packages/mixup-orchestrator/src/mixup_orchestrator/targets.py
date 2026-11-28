"""Docker execution: drive the Docker SDK from ``RunSpec``s on local and remote daemons.

The pure planning lives in ``deploy.py``; this module is the imperative layer that turns a
``RunSpec`` into a running container. It is deliberately thin so the policy can be
unit-tested without Docker, while keeping the executor straightforward.
"""

import logging

import docker
from docker.errors import ContainerError, ImageNotFound, NotFound
from docker.types import DeviceRequest

from mixup_core.constants import MODEL_NAME
from mixup_core.devices import GpuMode

from .deploy import CONTAINER_LEDGER_DIR, CONTAINER_MODELS_DIR, RunSpec

logger = logging.getLogger(__name__)

_CONTAINER_MODEL_FILE = f"{CONTAINER_MODELS_DIR}/{MODEL_NAME}/model.safetensors"


def get_client(docker_host: str) -> docker.DockerClient:
    """A Docker client for a node's daemon (``"local"``, ``ssh://...``, or ``tcp://...``)."""
    if docker_host in ("", "local"):
        return docker.from_env()
    return docker.DockerClient(base_url=docker_host)


def _device_requests(spec: RunSpec) -> list[DeviceRequest] | None:
    if spec.gpu is GpuMode.CDI:
        return [DeviceRequest(driver="cdi", device_ids=["nvidia.com/gpu=all"])]
    if spec.gpu is GpuMode.NVIDIA:
        return [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    return None


def _port_bindings(spec: RunSpec) -> dict:
    return {
        f"{p.container_port}/tcp": (p.bind_ip, p.host_port) if p.bind_ip else p.host_port
        for p in spec.ports
    }


def _volumes(spec: RunSpec) -> dict:
    return {v.source: {"bind": v.container_path, "mode": v.mode} for v in spec.volumes}


def _named_volume(spec: RunSpec, container_path: str) -> str | None:
    for volume in spec.volumes:
        if volume.container_path == container_path and volume.kind == "volume":
            return volume.source
    return None


def ensure_image(client: docker.DockerClient, spec: RunSpec) -> None:
    """Make sure the image is present, pulling it for remote daemons (or any missing ref)."""
    try:
        client.images.get(spec.image)
    except ImageNotFound:
        logger.info("PULLING %s", spec.image)
        client.images.pull(spec.image)


def _file_exists(
    client: docker.DockerClient, spec: RunSpec, volume: str, path: str
) -> bool:
    try:
        client.containers.run(
            spec.image,
            command=["test", "-f", path],
            volumes={volume: {"bind": CONTAINER_MODELS_DIR, "mode": "ro"}},
            remove=True,
        )
        return True
    except ContainerError:
        return False


def ensure_model(client: docker.DockerClient, spec: RunSpec) -> None:
    """Populate a remote node's named model volume (one-shot download) if it is empty.

    Local nodes bind the host ``models/`` directory, so there is nothing to do.
    """
    volume = _named_volume(spec, CONTAINER_MODELS_DIR)
    if volume is None:
        return
    if _file_exists(client, spec, volume, _CONTAINER_MODEL_FILE):
        logger.info("model present in volume %s", volume)
        return
    logger.info("downloading model into volume %s (this can take a while)", volume)
    client.containers.run(
        spec.image,
        command=["mixup-node-fetch"],  # default --dir = MODEL_DIR (the model-name subdir)
        volumes={volume: {"bind": CONTAINER_MODELS_DIR, "mode": "rw"}},
        remove=True,
        user="root",
    )


def ensure_ledger(client: docker.DockerClient, spec: RunSpec) -> None:
    """Make a remote node's named ledger volume writable by the ``app`` user."""
    volume = _named_volume(spec, CONTAINER_LEDGER_DIR)
    if volume is None:
        return
    client.containers.run(
        spec.image,
        command=["chown", "-R", "app:app", CONTAINER_LEDGER_DIR],
        volumes={volume: {"bind": CONTAINER_LEDGER_DIR, "mode": "rw"}},
        remove=True,
        user="root",
    )


def start(client: docker.DockerClient, spec: RunSpec) -> None:
    """Start the node container, recreating it if it is missing, stopped, or stale."""
    try:
        existing = client.containers.get(spec.container_name)
    except NotFound:
        existing = None

    if existing is not None:
        unchanged = existing.labels.get("mixup.config_hash") == spec.labels.get(
            "mixup.config_hash"
        )
        if existing.status == "running" and unchanged:
            logger.info("%s RUNNING (unchanged).", spec.container_name)
            return
        logger.info("%s stale or stopped; recreating.", spec.container_name)
        existing.remove(force=True)

    client.containers.run(
        image=spec.image,
        name=spec.container_name,
        detach=True,
        network=spec.network,
        network_mode=spec.network_mode,
        ports=_port_bindings(spec),
        restart_policy={"Name": spec.restart_policy},
        environment=spec.environment,
        volumes=_volumes(spec),
        device_requests=_device_requests(spec),
        nano_cpus=spec.nano_cpus,
        mem_limit=f"{spec.mem_limit_mb}m" if spec.mem_limit_mb else None,
        labels=spec.labels,
    )
    logger.info(
        "STARTED %s (gpu=%s, host=%s)", spec.container_name, spec.gpu.value, spec.docker_host
    )


def remove(client: docker.DockerClient, spec: RunSpec) -> None:
    """Stop and remove the node container if it exists."""
    try:
        container = client.containers.get(spec.container_name)
        container.remove(force=True)
        logger.info("REMOVED %s", spec.container_name)
    except NotFound:
        pass


def status(client: docker.DockerClient, spec: RunSpec) -> str:
    """The container's status string, or ``"not-found"``."""
    try:
        return client.containers.get(spec.container_name).status
    except NotFound:
        return "not-found"
