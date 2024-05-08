import torch as th

from torch.utils.data import Dataset
from typing import Sequence

from phynn.data import PhysicalData, PhysicsSampleKey


Tensors = Sequence[th.Tensor]


class PhynnDataset(Dataset):
    def __init__(
        self,
        data: PhysicalData,
        inputs: Sequence[PhysicsSampleKey],
        outputs: Sequence[PhysicsSampleKey],
    ) -> None:
        self._data = data

        self._in_keys = inputs
        self._out_keys = outputs

        if not data.has_params and (PhysicsSampleKey.PARAMS in (*inputs, *outputs)):
            raise ValueError(f"Provided data interface does not returns params.")

        self._series = data.times_shape[0]
        self._pairs_indices = {}

        series_length = data.times_shape[1]
        self._pairs = 0

        for t_start in range(series_length - 1):
            for t_end in range(t_start + 1, series_length):
                self._pairs_indices[self._pairs] = (t_start, t_end)
                self._pairs += 1

    def __len__(self) -> int:
        return self._series * self._pairs

    def __getitem__(self, index) -> tuple[Tensors, Tensors]:
        series = index // self._pairs
        pair = index % self._pairs
        t_start, t_end = self._pairs_indices[pair]

        sample = self._data.get(series, t_start, t_end)

        inputs = tuple(sample[element] for element in self._in_keys)
        outputs = tuple(sample[element] for element in self._out_keys)

        return inputs, outputs
