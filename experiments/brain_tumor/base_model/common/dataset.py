import torch as th
import torch.nn as nn
import torchvision.transforms as T

from pathlib import Path
from typing import Sequence

from phynn.data.img import (
    ImagesDataInterface,
    HDF5ImagesDataInterface,
    train_test_split,
)
from phynn.data.sim import DynamicSimulationDataset
from phynn.diff import DiffEquation
from phynn.diff.terms import DiffusionTerm, ProliferationTerm
from phynn.train import training_device


def get_data() -> tuple[ImagesDataInterface, ImagesDataInterface]:
    path = Path("./../data/processed/BRATS2020/result.h5")
    all = HDF5ImagesDataInterface(path, training_device)
    return train_test_split(all, 0.7)


class AugmentedDiffEquation(DiffEquation):
    def __init__(
        self, neural_eq_components: Sequence[nn.Module], threshold: float = 0.2
    ) -> None:
        super().__init__(neural_eq_components)
        self._threshold = threshold
        self._move = T.RandomAffine((-2, 2), (0.01, 0.01))

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        params = params * (1 + (th.rand_like(params) - 0.5) / 10)

        v = u - self._threshold
        v[v < 0] = 0
        v /= 1 - self._threshold
        diff = super().forward(v, params)
        diff[u < 0.05] = 0

        return self._move(diff)


def create_dataset(
    initial_conditions: ImagesDataInterface,
) -> DynamicSimulationDataset:
    diff_eq = AugmentedDiffEquation([DiffusionTerm(), ProliferationTerm()])
    diff_eq = diff_eq.to(training_device)

    params = lambda batch_size: (
        (th.rand((batch_size, 2)) * th.Tensor([[2.5, 2.5]])) ** 1.3
    ).to(training_device)

    return DynamicSimulationDataset(
        initial_conditions=initial_conditions,
        diff_eq=diff_eq,
        params_provider=params,
        max_simulation_steps=8,
        min_simulation_steps=3,
        max_pre_steps=8,
    )
