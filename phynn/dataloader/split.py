from typing import Sequence

from phynn.dataloader.base import ImageDynamics, DataInterface


class SplitDataInterface(DataInterface):
    def __init__(self, base_interface: DataInterface, start: int, end: int) -> None:
        self._base_interface = base_interface
        self._start = start
        self._end = end

    @property
    def image_shape(self) -> Sequence[int]:
        return self._base_interface.image_shape

    @property
    def times_shape(self) -> Sequence[int]:
        return self._end - self._start, *self._base_interface.times_shape[1:]

    def names(self, dim: int) -> Sequence[str]:
        return self._base_interface.names(dim)

    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        series = series + self._start
        return self._base_interface.get(series, t_start, t_end)
