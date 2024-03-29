import numpy as np

from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Sequence


class BaseDataLoader(Mapping[int, tuple[np.ndarray, float]], ABC):
    @property
    def images(self) -> Sequence[np.ndarray]:
        return [self[i][0] for i in range(len(self))]

    @property
    def time_elapsed(self) -> Sequence[float]:
        return [self[i][1] for i in range(len(self))]

    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        pass
