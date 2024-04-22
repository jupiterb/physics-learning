import torch as th

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Mapping, Sequence


class PhysicsSampleKey(Enum):
    START = 0
    RESULT = 1
    TIME_DIFF = 2
    PARAMS = 3


class PhysicsSample(Mapping[PhysicsSampleKey, th.Tensor]):
    def __init__(
        self,
        start: th.Tensor,
        result: th.Tensor,
        t_diff: th.Tensor,
        params: th.Tensor | None,
    ) -> None:
        self._mapping = {
            PhysicsSampleKey.START: start,
            PhysicsSampleKey.RESULT: result,
            PhysicsSampleKey.TIME_DIFF: t_diff,
        }

        if params is not None:
            self._mapping[PhysicsSampleKey.PARAMS] = params

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, element: PhysicsSampleKey) -> th.Tensor:
        return self._mapping[element]

    def __iter__(self) -> Iterator[PhysicsSampleKey]:
        return iter(self._mapping)


class PhysicalData(ABC):
    @property
    @abstractmethod
    def has_params(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def times_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get(self, series: int, t_start: int, t_end: int) -> PhysicsSample:
        raise NotImplementedError()
