import torch.nn as nn
import torch.optim as optim

from phynn.data.sim import DynamicSimulationDataset
from phynn.diff import DiffEquation
from phynn.models import DiffEquationModel, OptimizerParams
from phynn.train import train


def run_training(
    diff_eq_components_net: nn.Module,
    train_ds: DynamicSimulationDataset,
    test_ds: DynamicSimulationDataset,
    run_name: str,
    batch_size: int,
    epochs: int,
    lr: float = 0.00015,
) -> None:
    diff_eq_nn = DiffEquation(diff_eq_components_net, 2)

    diff_eq_model = DiffEquationModel(
        diff_eq_nn, optimizer_params=OptimizerParams(optim.AdamW, lr)
    )

    train(
        diff_eq_model,
        run_name=run_name,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=batch_size,
        epochs=epochs,
    )
