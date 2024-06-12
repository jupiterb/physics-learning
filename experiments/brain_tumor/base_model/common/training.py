import torch.nn as nn
import torch.optim as optim

from typing import Sequence

from phynn.data.sim import DynamicSimulationDataset
from phynn.diff import DiffEquation
from phynn.models import DiffEquationModel, OptimizerParams
from phynn.train import train


_INPUT_SHAPE = (1, 120, 120)


def run_training(
    neural_nets: Sequence[nn.Module],
    train_ds: DynamicSimulationDataset,
    test_ds: DynamicSimulationDataset,
    run_name: str,
    epochs: int,
    lr: float = 0.00015,
) -> None:
    diff_eq_nn = DiffEquation(neural_nets)

    diff_eq_model = DiffEquationModel(
        diff_eq_nn, _INPUT_SHAPE, optimizer_params=OptimizerParams(optim.Adam, lr)
    )

    train(
        diff_eq_model,
        run_name=run_name,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=32,
        epochs=epochs,
    )
