from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


class BaseDataFinder(ABC):
    @abstractmethod
    def find(self) -> Sequence[tuple[tuple[str, ...], Path]]:
        raise NotImplementedError()
