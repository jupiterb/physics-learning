from abc import ABC, abstractmethod
import h5py
import os
import torch as th
from typing import Optional, Callable, Sequence, Union

from phynn.data.utils import HDF5DataExportManager


Indices = Union[Sequence[int], th.Tensor]
SpatioTemporalDataSample = tuple[
    th.Tensor,  # image
    th.Tensor,  # time
]


class TimeSeriesDataInterface(ABC):
    @property
    @abstractmethod
    def series_number(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def series_length(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_batch(
        self, series_ixs: Indices, time_ixs: Indices
    ) -> SpatioTemporalDataSample:
        raise NotImplementedError()


class TimeSeriesDataInterfaceWrapper(TimeSeriesDataInterface):
    def __init__(
        self, data: TimeSeriesDataInterface, func: Callable[[th.Tensor], th.Tensor]
    ) -> None:
        self._data = data
        self._func = func

    @property
    def series_number(self) -> int:
        return self._data.series_number

    @property
    def series_length(self) -> int:
        return self._data.series_length

    @property
    def image_shape(self) -> Sequence[int]:
        return self._data.image_shape

    def get_batch(
        self, series_ixs: Indices, time_ixs: Indices
    ) -> SpatioTemporalDataSample:
        images, times = self._data.get_batch(series_ixs, time_ixs)
        return self._func(images), times


class HDF5TimeSeriesDataInterface(TimeSeriesDataInterface):
    def __init__(
        self,
        path: os.PathLike,
        device: th.device,
        data_type: th.dtype = th.float32,
    ) -> None:
        with h5py.File(path, "r") as file:
            self._images = th.tensor(file["images"][:], dtype=data_type).to(device)  # type: ignore
            self._times = th.tensor(file["times"][:], dtype=data_type).to(device)  # type: ignore

    @property
    def series_number(self) -> int:
        return self._images.shape[0]

    @property
    def series_length(self) -> int:
        return self._images.shape[1]

    @property
    def image_shape(self) -> Sequence[int]:
        return self._images.shape[2:]

    def get_batch(
        self, series_ixs: Indices, time_ixs: Indices
    ) -> SpatioTemporalDataSample:
        return self._images[series_ixs][time_ixs], self._times[series_ixs][time_ixs]


class TimeSeriesDataVariationInterface(TimeSeriesDataInterface):
    def __init__(self, data: TimeSeriesDataInterface, variation: th.Tensor) -> None:
        self._data = data
        self._variation = variation

    @property
    def series_number(self) -> int:
        return len(self._variation)

    @property
    def series_length(self) -> int:
        return self._data.series_length

    @property
    def image_shape(self) -> Sequence[int]:
        return self._data.image_shape

    def get_batch(
        self, series_ixs: Indices, time_ixs: Indices
    ) -> SpatioTemporalDataSample:
        series_ixs = self._variation[series_ixs]
        return self._data.get_batch(series_ixs, time_ixs)


def train_test_split(
    data: TimeSeriesDataInterface, train_factor: float
) -> tuple[TimeSeriesDataInterface, TimeSeriesDataInterface]:
    train_size = int(data.series_number * train_factor)
    permutation = th.randperm(data.series_number)

    train_variation = permutation[:train_size]
    test_variation = permutation[train_size:]

    train_data = TimeSeriesDataVariationInterface(data, train_variation)
    test_data = TimeSeriesDataVariationInterface(data, test_variation)

    return train_data, test_data


def save(
    data: TimeSeriesDataInterface,
    path: os.PathLike,
    batch_size: Optional[int] = None,
    override: bool = True,
):
    batch_size = data.series_length if batch_size is None else batch_size
    time_ixs = list(range(data.series_length))

    with HDF5DataExportManager(path, override) as export:
        images = export.dataset("images", data.image_shape)
        times = export.dataset("times", data.image_shape)

        for start in range(0, data.series_length, batch_size):
            stop = min(data.series_length, start + batch_size)
            series_ixs = list(range(start, stop))

            images_batch, times_batch = data.get_batch(series_ixs, time_ixs)

            images.append(images_batch)
            times.append(times_batch)
