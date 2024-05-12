from datetime import datetime
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Sequence
import wandb

from phynn.models import BaseModel


def train(
    model: BaseModel,
    train_datasets: Sequence[Dataset],
    val_datasets: Sequence[Dataset],
    batch_size: int,
    epochs: int,
) -> BaseModel:
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    run_name = f"{model.name}_{datetime.now().strftime('%Y.%m.%d_%H:%M')}"
    logger = WandbLogger(project="physics-learning", name=run_name)

    try:
        trainer = L.Trainer(max_epochs=epochs, logger=logger)
        trainer.fit(model, train_dataloader, val_dataloader)
        wandb.finish()
    except Exception as e:
        print(f"Exception raised: {e}")
        wandb.finish(1)

    return model
