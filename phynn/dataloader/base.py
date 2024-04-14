import torch as th

from abc import ABC, abstractmethod
from typing import Iterator, Mapping, Sequence


ImageDynamicsNoParams = tuple[tuple[th.Tensor, float], th.Tensor]
ImageDynamicsWithParams = tuple[tuple[th.Tensor, float, th.Tensor], th.Tensor]

ImageDynamics = ImageDynamicsNoParams | ImageDynamicsWithParams


class DataInterface(ABC):
    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def times_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        raise NotImplementedError()


class DataLoader(Mapping[int, ImageDynamics]):
    def __init__(self, data_interface: DataInterface) -> None:
        self._data_interface = data_interface
        self._series = self._data_interface.times_shape[0]
        self._intervals = self._data_interface.times_shape[1] - 1

    @property
    def shape(self) -> Sequence[int]:
        return self._data_interface.image_shape

    def __len__(self) -> int:
        return self._series * self._intervals

    def __getitem__(self, index: int) -> ImageDynamics:
        series = index // self._intervals
        t_start = index % self._intervals
        t_end = t_start + 1
        return self._data_interface.get(series, t_start, t_end)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))
