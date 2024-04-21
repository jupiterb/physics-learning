import h5py
import os
import torch as th

from typing import Sequence

from phynn import default_device
from phynn.dataloader.base import DynamicsSample, DataInterface


def _get_params(file: h5py.File, device: th.device) -> th.Tensor | None:
    if "params" in file:
        return th.tensor(file["params"][:], dtype=th.float32).to(device)  # type: ignore
    else:
        return None


class HDF5DirectlyFromFile(DataInterface):
    def __init__(self, path: os.PathLike, device: th.device = default_device) -> None:
        self._path = path
        self._device = device

        with h5py.File(self._path, "r") as file:
            self._shape = file["images"][:].shape  # type: ignore

    @property
    def has_params(self) -> bool:
        with h5py.File(self._path, "r") as file:
            return _get_params(file, self._device) is not None

    @property
    def image_shape(self) -> Sequence[int]:
        return self._shape[2:]

    @property
    def times_shape(self) -> Sequence[int]:
        return self._shape[:2]

    def get(self, series: int, t_start: int, t_end: int) -> DynamicsSample:
        with h5py.File(self._path, "r") as file:
            start = th.tensor(file["images"][series][t_start], dtype=th.float32).to(self._device)  # type: ignore
            result = th.tensor(file["images"][series][t_end], dtype=th.float32).to(self._device)  # type: ignore
            times = th.tensor(file["times"][series], dtype=th.int32).to(self._device)  # type: ignore
            time_diff = (times[t_end] - times[t_start]).unsqueeze(0)
            params = _get_params(file, self._device)

            return DynamicsSample(start, result, time_diff, params)


class HDF5LoadToMemory(DataInterface):
    def __init__(self, path: os.PathLike, device: th.device = default_device) -> None:
        with h5py.File(path, "r") as file:
            self._images = th.tensor(file["images"][:], dtype=th.float32).to(device)  # type: ignore
            self._times = th.tensor(file["times"][:], dtype=th.int32).to(device)  # type: ignore
            self._params = _get_params(file, device)

    @property
    def has_params(self) -> bool:
        return self._params is not None

    @property
    def image_shape(self) -> Sequence[int]:
        return self._images.shape[2:]

    @property
    def times_shape(self) -> Sequence[int]:
        return self._times.shape

    def get(self, series: int, t_start: int, t_end: int) -> DynamicsSample:
        start = self._images[series][t_start]
        result = self._images[series][t_end]

        time_diff = (
            self._times[series][t_end] - self._times[series][t_start]
        ).unsqueeze(0)

        params = None if self._params is None else self._params[series]

        return DynamicsSample(start, result, time_diff, params)
