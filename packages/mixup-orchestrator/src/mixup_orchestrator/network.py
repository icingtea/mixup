"""Docker network management for the lab (``mixup-lab``)."""

import logging

import docker
from docker.errors import NotFound
from docker.models.networks import Network

from mixup_core.constants import NETWORK_NAME

logger = logging.getLogger(__name__)


def get_or_create_network(client: docker.DockerClient) -> Network:
    """Return the cluster network, creating it if it does not exist."""
    try:
        network = client.networks.get(NETWORK_NAME)
        logger.info("FOUND NETWORK: %s", NETWORK_NAME)
    except NotFound:
        network = client.networks.create(name=NETWORK_NAME, driver="bridge")
        logger.info("CREATED NETWORK: %s", NETWORK_NAME)
    return network
