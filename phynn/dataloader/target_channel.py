from typing import Sequence

from phynn.dataloader.base import ImageDynamics, DataInterface


class TargetChannelDataInterface(DataInterface):
    def __init__(self, base_interface: DataInterface, target_channel: int) -> None:
        self._base_interface = base_interface
        self._target_channel = target_channel

    @property
    def image_shape(self) -> Sequence[int]:
        return self._base_interface.image_shape

    @property
    def times_shape(self) -> Sequence[int]:
        return self._base_interface.times_shape

    def names(self, dim: int) -> Sequence[str]:
        return self._base_interface.names(dim)

    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        start, end, time_diff = self._base_interface.get(series, t_start, t_end)
        return start, end[self._target_channel].unsqueeze(0), time_diff
