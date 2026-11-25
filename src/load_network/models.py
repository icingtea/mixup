"""The shape of src/config.json: which container is central and which experts each other container holds."""

from pydantic import BaseModel
from typing import Dict, List, TypeAlias


ContainerName: TypeAlias = str
"""A container's name, also its hostname on the Docker network, e.g. "container_2"."""

ExpertList: TypeAlias = List[int]
"""Expert numbers (0-31) held by one container."""

ExpertAlloc: TypeAlias = Dict[ContainerName, ExpertList]
"""Every expert container and the experts it holds."""


class Config(BaseModel):
    """The cluster layout read from src/config.json."""

    central: ContainerName
    """The container that runs everything except the experts."""

    experts: ExpertAlloc
    """Expert container name -> the expert numbers it holds, e.g. {"container_2": [0, ..., 7]}."""
