import torch as th
import torch.optim as optim
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
from phynn.models import DiffEquationModel, OptimizerParams
from phynn.nn import ResBlock, Conv, ConvBlockParams
from phynn.train import train, training_device


def get_data() -> tuple[ImagesDataInterface, ImagesDataInterface]:
    path = Path("./data/processed/BRATS2020/result.h5")
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


def _create_u_net() -> nn.Module:
    inner = (
        Conv(48)
        .append(ConvBlockParams(64, 3, rescale=2))
        .transpose(True)  # type: ignore
        .append(ConvBlockParams(48, 3, rescale=2))
    )
    inner_res = ResBlock(inner.unload())

    down_sampling = Conv(32).append(ConvBlockParams(48, 3, rescale=2, dropout=0.1))
    up_sampling = Conv(32, transpose=True).prepend(ConvBlockParams(48, 3, rescale=2))

    middle = ResBlock(
        nn.Sequential(down_sampling.unload(), inner_res, up_sampling.unload())
    )

    down_sampling.prepend(ConvBlockParams(1, 3, rescale=2, dropout=0.3))
    up_sampling.append(ConvBlockParams(1, 3, nn.Hardtanh, rescale=2))

    return ResBlock(
        nn.Sequential(down_sampling.unload(), middle, up_sampling.unload())
    ).to(training_device)


def create_model() -> DiffEquationModel:
    diff_eq_nn = DiffEquation([_create_u_net(), _create_u_net()])
    diff_eq_model = DiffEquationModel(
        diff_eq_nn, optimizer_params=OptimizerParams(optim.Adam, 0.00015)
    )
    return diff_eq_model


def run_training(
    model: DiffEquationModel,
    train_ds: DynamicSimulationDataset,
    test_ds: DynamicSimulationDataset,
) -> None:
    train(
        model,
        run_name="equation_diffusion_proliferation_together",
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=64,
        epochs=50,
    )


def main() -> None:
    train_ics, test_ics = get_data()
    train_ds = create_dataset(train_ics)
    test_ds = create_dataset(test_ics)
    model = create_model()
    run_training(model, train_ds, test_ds)


if __name__ == "__main__":
    main()
