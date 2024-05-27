from __future__ import annotations

from contextlib import AbstractContextManager
import h5py
import os
import torch as th
from types import TracebackType
from typing import Generator, Sequence


class HDF5DataSetExporter:
    def __init__(self, dataset: h5py.Dataset) -> None:
        self._dataset = dataset

    def put(self, batch: th.Tensor) -> None:
        batch_size = batch.shape[0]
        current_size = self._dataset.shape[0]
        new_size = current_size + batch_size
        self._dataset.resize(new_size, axis=0)
        self._dataset[current_size:new_size, ...] = batch.detach().numpy()


class HDF5DataExportManager(AbstractContextManager):
    def __init__(self, path: os.PathLike, override: bool) -> None:
        if os.path.exists(path) and not override:
            raise FileExistsError(
                f"The file {path} already exists and override is set to False."
            )

        self._path = path

    def __enter__(self) -> HDF5DataExportManager:
        self._file = h5py.File(self._path, "w")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._file.close()

    def dataset(self, name: str, data_shape: Sequence[int]) -> HDF5DataSetExporter:
        dataset = self._file.create_dataset(
            name,
            shape=(0, *data_shape),
            maxshape=(None, *data_shape),
            dtype="float32",
        )
        return HDF5DataSetExporter(dataset)
