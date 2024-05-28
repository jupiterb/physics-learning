from abc import ABC, abstractmethod
import h5py
import os
import torch as th
from typing import Callable, Sequence

from phynn.data.utils import HDF5DataExportManager


Indices = Sequence[int] | th.Tensor


class ImagesDataInterface(ABC):
    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_batch(self, ixs: Indices) -> th.Tensor:
        raise NotImplementedError()


class ImagesDataInterfaceWrapper(ImagesDataInterface):
    def __init__(
        self, data: ImagesDataInterface, func: Callable[[th.Tensor], th.Tensor]
    ) -> None:
        self._data = data
        self._func = func

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def image_shape(self) -> Sequence[int]:
        return self._func(self.get_batch([0])).shape

    def get_batch(self, ixs: Indices) -> th.Tensor:
        return self._func(self._data.get_batch(ixs))


class HDF5ImagesDataInterface(ImagesDataInterface):
    def __init__(
        self,
        path: os.PathLike,
        device: th.device,
        data_type: th.dtype = th.float32,
    ) -> None:
        with h5py.File(path, "r") as file:
            self._data = th.tensor(file["data"][:], dtype=data_type).to(device)  # type: ignore

    @property
    def size(self) -> int:
        return self._data.shape[0]

    @property
    def image_shape(self) -> Sequence[int]:
        return self._data.shape[1:]

    def get_batch(self, ixs: Indices) -> th.Tensor:
        return self._data[ixs]


class ImagesDataVariationInterface(ImagesDataInterface):
    def __init__(self, data: ImagesDataInterface, variation: th.Tensor) -> None:
        self._data = data
        self._variation = variation

    @property
    def size(self) -> int:
        return len(self._variation)

    @property
    def image_shape(self) -> Sequence[int]:
        return self._data.image_shape

    def get_batch(self, ixs: Indices) -> th.Tensor:
        ixs = self._variation[ixs]
        return self._data.get_batch(ixs)


def train_test_split(
    data: ImagesDataInterface, train_factor: float
) -> tuple[ImagesDataInterface, ImagesDataInterface]:
    train_size = int(data.size * train_factor)
    permutation = th.randperm(data.size)

    train_variation = permutation[:train_size]
    test_variation = permutation[train_size:]

    train_data = ImagesDataVariationInterface(data, train_variation)
    test_data = ImagesDataVariationInterface(data, test_variation)

    return train_data, test_data


def save(
    data: ImagesDataInterface,
    path: os.PathLike,
    batch_size: int | None = None,
    override: bool = True,
):
    batch_size = data.size if batch_size is None else batch_size

    with HDF5DataExportManager(path, override) as export:
        dataset = export.dataset("data", data.image_shape)

        for start in range(0, data.size, batch_size):
            stop = min(data.size, start + batch_size)
            ixs = list(range(start, stop))
            batch = data.get_batch(ixs)
            dataset.append(batch)
