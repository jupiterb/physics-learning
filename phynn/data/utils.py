from random import shuffle
from typing import Sequence

from phynn.data.base import PhysicsSample, PhysicalData


class ShufflePhysicalData(PhysicalData):
    def __init__(self, base_interface: PhysicalData) -> None:
        self._base_interface = base_interface
        self._shuffle = list(range(self.times_shape[0]))
        shuffle(self._shuffle)

    @property
    def has_params(self) -> bool:
        return self._base_interface.has_params

    @property
    def image_shape(self) -> Sequence[int]:
        return self._base_interface.image_shape

    @property
    def times_shape(self) -> Sequence[int]:
        return self._base_interface.times_shape

    def get(self, series: int, t_start: int, t_end: int) -> PhysicsSample:
        series = self._shuffle[series]
        return self._base_interface.get(series, t_start, t_end)


class SplitPhysicalData(PhysicalData):
    def __init__(self, base_interface: PhysicalData, start: int, end: int) -> None:
        self._base_interface = base_interface
        self._start = start
        self._end = end

    @property
    def has_params(self) -> bool:
        return self._base_interface.has_params

    @property
    def image_shape(self) -> Sequence[int]:
        return self._base_interface.image_shape

    @property
    def times_shape(self) -> Sequence[int]:
        return self._end - self._start, *self._base_interface.times_shape[1:]

    def get(self, series: int, t_start: int, t_end: int) -> PhysicsSample:
        series = series + self._start
        return self._base_interface.get(series, t_start, t_end)
