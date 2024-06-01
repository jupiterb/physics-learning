from datetime import datetime
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import wandb

from phynn.models import BaseModel


def train(
    model: BaseModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    epochs: int,
) -> BaseModel:
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    run_name = f"{model.name}_{datetime.now().strftime('%Y.%m.%d_%H:%M')}"
    logger = WandbLogger(project="physics-learning", name=run_name)

    try:
        trainer = L.Trainer(max_epochs=epochs, logger=logger, log_every_n_steps=1)
        trainer.fit(model, train_dataloader, val_dataloader)
        wandb.finish()
    except Exception as e:
        print(f"Exception raised: {e}")
        wandb.finish(1)
        raise e

    return model
