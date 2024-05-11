import h5py
import os
import torch as th
from typing import Sequence

from phynn.data.interface.base import DataInterface, SampleIndex


class HDF5ToMemory(DataInterface):
    def __init__(
        self,
        path: os.PathLike,
        key: str,
        device: th.device,
        data_type: th.dtype = th.float32,
    ) -> None:
        with h5py.File(path, "r") as file:
            self._data = th.tensor(file[key][:], dtype=data_type).to(device)  # type: ignore

    @property
    def series_number(self) -> int:
        return self._data.shape[0]

    @property
    def series_length(self) -> int:
        return self._data.shape[1]

    @property
    def sample_shape(self) -> Sequence[int]:
        return self._data.shape[2:]

    def get(self, series_ixs: SampleIndex, time_ixs: SampleIndex) -> th.Tensor:
        return self._data[series_ixs][time_ixs]
