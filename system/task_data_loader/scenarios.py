from dataclasses import dataclass, field
from typing import List, Callable, Optional, Union
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

@dataclass
class TaskConfig:
    train: Dataset
    test: Dataset
    id: str
    nb_classes: int


class Scenario(ABC):
    def __init__(self, root: str = "", transforms: Optional[Union[List[Callable], Callable]] = None):
        self.root = root
        self.transforms = transforms

    @property
    @abstractmethod
    def tasks(self) -> List[TaskConfig]:
        pass


class SimpleScenario(Scenario):
    """Concrete Scenario that wraps a pre-built list of TaskConfigs."""

    def __init__(self, tasks: List[TaskConfig], root: str = "", transforms=None):
        super().__init__(root=root, transforms=transforms)
        self._tasks = tasks

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks