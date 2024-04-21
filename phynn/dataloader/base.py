import torch as th

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Mapping, Sequence


class DynamicsElement(Enum):
    START = 0
    RESULT = 1
    TIME_DIFF = 2
    PARAMS = 3


class DynamicsSample(Mapping[DynamicsElement, th.Tensor]):
    def __init__(
        self,
        start: th.Tensor,
        result: th.Tensor,
        t_diff: th.Tensor,
        params: th.Tensor | None,
    ) -> None:
        self._mapping = {
            DynamicsElement.START: start,
            DynamicsElement.RESULT: result,
            DynamicsElement.TIME_DIFF: t_diff,
        }

        if params is not None:
            self._mapping[DynamicsElement.PARAMS] = params

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, element: DynamicsElement) -> th.Tensor:
        return self._mapping[element]

    def __iter__(self) -> Iterator[DynamicsElement]:
        return iter(self._mapping)


class DataInterface(ABC):
    @property
    @abstractmethod
    def has_params(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def times_shape(self) -> Sequence[int]:
        raise NotImplementedError()

    @abstractmethod
    def get(self, series: int, t_start: int, t_end: int) -> DynamicsSample:
        raise NotImplementedError()


Tensors = Sequence[th.Tensor]


class DataLoader(Mapping[int, tuple[Tensors, Tensors]]):
    def __init__(
        self,
        data_interface: DataInterface,
        input_elements: Sequence[DynamicsElement],
        output_elements: Sequence[DynamicsElement],
    ) -> None:
        self._data_interface = data_interface

        self._series = self._data_interface.times_shape[0]
        self._intervals = self._data_interface.times_shape[1] - 1

        self._input_elements = input_elements
        self._output_elements = output_elements

        if not self._data_interface.has_params and (
            DynamicsElement.PARAMS in input_elements
            or DynamicsElement.PARAMS in output_elements
        ):
            raise ValueError(f"Provided data interface does not returns params.")

    @property
    def shape(self) -> Sequence[int]:
        return self._data_interface.image_shape

    def __len__(self) -> int:
        return self._series * self._intervals

    def __getitem__(self, index: int) -> tuple[Tensors, Tensors]:
        series = index // self._intervals
        t_start = index % self._intervals
        t_end = t_start + 1

        sample = self._data_interface.get(series, t_start, t_end)

        inputs = tuple(sample[element] for element in self._input_elements)
        outputs = tuple(sample[element] for element in self._output_elements)

        return inputs, outputs

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))
