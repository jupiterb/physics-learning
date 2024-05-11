from abc import ABC, abstractmethod
import torch as th
from torch.utils.data import Dataset
from typing import Sequence

from phynn.data.interface import DataInterface


class FlatDataset(Dataset):
    def __init__(self, data: DataInterface) -> None:
        self._data = data

    def __len__(self) -> int:
        return self._data.series_number * self._data.series_length

    def __getitem__(self, index: int) -> th.Tensor:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        tensor_indices: th.Tensor = th.Tensor(indices).type(th.int)
        series_ixs = tensor_indices // self._data.series_length
        time_ixs = tensor_indices % self._data.series_length
        return self._data.get(series_ixs, time_ixs)


class SimulationDataIndexer:
    def __init__(self, series_number: int, series_length: int) -> None:
        self._series_number = series_number
        self._series_samples = (series_length - 1) * series_length // 2

        start_ixs = []
        result_ixs = []

        for i in range(series_length - 1):
            for j in range(i, series_length):
                start_ixs.append(i)
                result_ixs.append(j)

        self._start_ixs = th.Tensor(start_ixs).type(th.int)
        self._result_ixs = th.Tensor(result_ixs).type(th.int)

    @property
    def size(self) -> int:
        return self._series_number * self._series_samples

    def simulation_indices(
        self, indices: Sequence[int]
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        tensor_indices: th.Tensor = th.Tensor(indices)
        series_ixs = tensor_indices // self._series_samples
        sample_ixs = tensor_indices % self._series_samples
        return series_ixs, self._start_ixs[sample_ixs], self._result_ixs[sample_ixs]


class _SimulationDataset(Dataset, ABC):
    def __init__(self, data: DataInterface, indexer: SimulationDataIndexer) -> None:
        self._data = data
        self._indexer = indexer

    def __len__(self) -> int:
        return self._indexer.size

    def __getitem__(self, index: int) -> th.Tensor:
        return self.__getitems__([index])[0]

    @abstractmethod
    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        raise NotImplementedError()


class SimulationStartDataset(_SimulationDataset):
    def __init__(
        self, spatial_data: DataInterface, indexer: SimulationDataIndexer
    ) -> None:
        super().__init__(spatial_data, indexer)

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        series_ixs, start_ixs, _ = self._indexer.simulation_indices(indices)
        return self._data.get(series_ixs, start_ixs)


class SimulationResultDataset(_SimulationDataset):
    def __init__(self, data: DataInterface, indexer: SimulationDataIndexer) -> None:
        super().__init__(data, indexer)

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        series_ixs, _, result_ixs = self._indexer.simulation_indices(indices)
        return self._data.get(series_ixs, result_ixs)


class SimulationDurationDataset(_SimulationDataset):
    def __init__(self, data: DataInterface, indexer: SimulationDataIndexer) -> None:
        super().__init__(data, indexer)

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        series_ixs, start_ixs, result_ixs = self._indexer.simulation_indices(indices)
        start = self._data.get(series_ixs, start_ixs)
        result = self._data.get(series_ixs, result_ixs)
        return result - start


class SimulationParamsDataset(_SimulationDataset):
    def __init__(self, data: DataInterface, indexer: SimulationDataIndexer) -> None:
        super().__init__(data, indexer)

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        series_ixs, _, _ = self._indexer.simulation_indices(indices)
        return self._data.get(series_ixs, th.zeros_like(series_ixs).type(th.int))
