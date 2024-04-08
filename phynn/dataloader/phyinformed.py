import torch as th

from typing import Sequence

from phynn.dataloader.base import ImageDynamics, DataInterface
from phynn.pde import PDEEval


class PhysicsInformedDataInterface(DataInterface):
    def __init__(
        self, base_interface: DataInterface, physics_eval: PDEEval, physics_w: float
    ) -> None:
        self._base_interface = base_interface
        self._physics_eval = physics_eval
        self._physics_w = physics_w

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

        with th.no_grad():
            end = end.clone() * (1 - self._physics_w)
            end += self._physics_eval(start, time_diff) * self._physics_w

        return start, end, time_diff
