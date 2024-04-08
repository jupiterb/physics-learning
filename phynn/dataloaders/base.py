import numpy as np

from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Sequence


ImageDynamics = tuple[np.ndarray, np.ndarray, float]


class BaseDataLoader(Mapping[int, ImageDynamics], ABC):
    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> ImageDynamics:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError()
