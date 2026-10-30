from pydantic import BaseModel
from typing import Dict, List, TypeAlias


ContainerName: TypeAlias = str
ExpertList: TypeAlias = List[int]
ExpertAlloc: TypeAlias = Dict[ContainerName, ExpertList]

class Config(BaseModel):
    central: ContainerName
    experts: ExpertAlloc