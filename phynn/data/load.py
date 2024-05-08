import h5py
import os
import torch as th

from typing import Sequence

from phynn import default_device
from phynn.data.base import PhysicsSample, PhysicalData


class HDF5LoadToMemory(PhysicalData):
    def __init__(self, path: os.PathLike, device: th.device = default_device) -> None:
        with h5py.File(path, "r") as file:
            self._images = th.tensor(file["images"][:], dtype=th.float32).to(device)  # type: ignore
            self._times = th.tensor(file["times"][:], dtype=th.int32).to(device)  # type: ignore

            if "params" in file:
                self._params = th.tensor(file["params"][:], dtype=th.float32).to(device)  # type: ignore
            else:
                self._params = None

    @property
    def has_params(self) -> bool:
        return self._params is not None

    @property
    def image_shape(self) -> Sequence[int]:
        return self._images.shape[2:]

    @property
    def times_shape(self) -> Sequence[int]:
        return self._times.shape

    def get(self, series: int, t_start: int, t_end: int) -> PhysicsSample:
        start = self._images[series][t_start]
        result = self._images[series][t_end]

        time_diff = (
            self._times[series][t_end] - self._times[series][t_start]
        ).unsqueeze(0)

        params = None if self._params is None else self._params[series]

        return PhysicsSample(start, result, time_diff, params)
