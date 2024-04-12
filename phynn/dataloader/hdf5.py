import h5py
import os
import torch as th

from typing import Sequence

from phynn import device
from phynn.dataloader.base import ImageDynamics, DataInterface


class HDF5DirectlyFromFile(DataInterface):
    def __init__(self, path: os.PathLike) -> None:
        self._path = path

        with h5py.File(self._path, "r") as file:
            self._shape = file["images"][:].shape  # type: ignore

    @property
    def image_shape(self) -> Sequence[int]:
        return self._shape[2:]

    @property
    def times_shape(self) -> Sequence[int]:
        return self._shape[:2]

    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        with h5py.File(self._path, "r") as file:
            start = th.tensor(file["images"][series][t_start], dtype=th.float32).to(device)  # type: ignore
            end = th.tensor(file["images"][series][t_end], dtype=th.float32).to(device)  # type: ignore
            times = th.tensor(file["times"][series], dtype=th.int32).to(device)  # type: ignore
            time_diff = (times[t_end] - times[t_start]).item()
            return start, end, time_diff


class HDF5LoadToMemory(DataInterface):
    def __init__(self, path: os.PathLike) -> None:
        with h5py.File(path, "r") as file:
            self._images = th.tensor(file["images"][:], dtype=th.float32).to(device)  # type: ignore
            self._times = th.tensor(file["times"][:], dtype=th.int32).to(device)  # type: ignore

    @property
    def image_shape(self) -> Sequence[int]:
        return self._images.shape[2:]

    @property
    def times_shape(self) -> Sequence[int]:
        return self._times.shape

    def get(self, series: int, t_start: int, t_end: int) -> ImageDynamics:
        start = self._images[series][t_start]
        end = self._images[series][t_end]
        time_diff = (self._times[series][t_end] - self._times[series][t_start]).item()
        return start, end, time_diff
