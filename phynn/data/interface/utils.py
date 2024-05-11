import torch as th
from typing import Sequence

from phynn.data.interface.base import DataInterface, SampleIndex


class DataVariation(DataInterface):
    def __init__(self, data: DataInterface, variation: th.Tensor) -> None:
        self._data = data
        self._variation = variation

    @property
    def series_number(self) -> int:
        return len(self._variation)

    @property
    def series_length(self) -> int:
        return self._data.series_length

    @property
    def sample_shape(self) -> Sequence[int]:
        return self._data.sample_shape

    def get(self, series_ixs: SampleIndex, time_ixs: SampleIndex) -> th.Tensor:
        series_ixs = self._variation[series_ixs]
        return self._data.get(series_ixs, time_ixs)


class TrainTestData:
    def __init__(self, series_number: int, train_factor: float) -> None:
        self._series_number = series_number

        train_size = int(series_number * train_factor)
        permutation = th.randperm(series_number)

        self._train_variation = permutation[:train_size]
        self._test_variation = permutation[train_size:]

    def split(self, data: DataInterface) -> tuple[DataInterface, DataInterface]:
        if data.series_number != self._series_number:
            raise ValueError(
                f"Series number of given DataInterface should be {self._series_number}, but is {data.series_number}"
            )

        train_data = DataVariation(data, self._train_variation)
        test_data = DataVariation(data, self._test_variation)

        return train_data, test_data
