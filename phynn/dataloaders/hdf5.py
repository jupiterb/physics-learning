import h5py
import os
import numpy as np

from typing import Iterator, Sequence

from phynn.dataloaders.base import ImageDynamics, BaseDataLoader


class HDF5DataLoader(BaseDataLoader):
    def __init__(self, path: os.PathLike) -> None:
        self._path = path
        self._load_data()

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    def names(self, dim: int) -> Sequence[str]:
        return self._dim_names[dim]

    def _load_data(self):
        self._dim_names = {}

        with h5py.File(self._path, "r") as file:
            self._shape = file["images"].shape[2:]  # type: ignore
            self._times: np.ndarray = file["times"].values  # type: ignore

            self._series = self._times.shape[0]  # type: ignore
            self._intervals = self._times.shape[1] - 1  # type: ignore

            for dim in file.keys():
                if dim.isdigit():
                    dim_names = [name.decode("utf-8") for name in file[dim]["names"][:]]  # type: ignore
                    self._dim_names[int(dim)] = dim_names

    def __len__(self) -> int:
        return self._series * self._intervals

    def __getitem__(self, index: int) -> ImageDynamics:
        with h5py.File(self._path, "r") as file:
            interval = index % self._intervals
            series = index // self._intervals

            start: np.ndarray = file["images"][series][interval]  # type: ignore
            end: np.ndarray = file["images"][series][interval + 1]  # type: ignore
            time_diff = (
                self._times[series][interval] - self._times[series][interval - 1]
            ).item()

        return start, end, time_diff

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))
