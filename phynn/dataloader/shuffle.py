from random import shuffle
from typing import Sequence

from phynn.dataloader.base import ImageDynamics, DataInterface


class ShuffleDataInterface(DataInterface):
    def __init__(self, base_interface: DataInterface) -> None:
        self._base_interface = base_interface
        self._shuffle = list(range(self.times_shape[0]))
        shuffle(self._shuffle)

    @property
    def image_shape(self) -> Sequence[int]:
        return self._base_interface.image_shape

    @property
    def times_shape(self) -> Sequence[int]:
        return self._base_interface.times_shape

    def names(self, dim: int) -> Sequence[str]:
        return self._base_interface.names(dim)

    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        series = self._shuffle[series]
        return self._base_interface.get(series, t_start, t_end)
