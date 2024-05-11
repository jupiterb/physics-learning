from abc import ABC, abstractmethod
import torch as th
from typing import Sequence


SampleIndex = int | Sequence[int] | th.Tensor


class DataInterface(ABC):
    @property
    @abstractmethod
    def series_number(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def series_length(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sample_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get(self, series_ixs: SampleIndex, time_ixs: SampleIndex) -> th.Tensor:
        raise NotImplementedError()
