import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import wandb

from phynn.models import BaseModel


def train(
    model: BaseModel,
    run_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    epochs: int,
) -> BaseModel:
    train_dataloader = DataLoader(train_dataset, batch_size, True)
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")

    logger = WandbLogger(project="physics-learning", name=run_name, log_model="all")

    try:
        trainer = L.Trainer(
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        wandb.finish()
    except Exception as e:
        print(f"Exception raised: {e}")
        wandb.finish(1)
        raise e

    return model
