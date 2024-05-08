import torch as th

from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Type
from tqdm import tqdm

from phynn.train.dataset import PhynnDataset


OptimizerClass = (
    Type[optim.Adam] | Type[optim.AdamW] | Type[optim.SGD] | Type[optim.RMSprop]
)


class Trainer:
    def __init__(
        self,
        optimizer_cls: OptimizerClass,
        loss_fun: nn.Module,
        lr: float,
        batch_size: int,
    ) -> None:
        self._optimizer_cls = optimizer_cls
        self._lr = lr
        self._batch_size = batch_size
        self._loss_fun = loss_fun

    def run(
        self,
        model: nn.Module,
        training_ds: PhynnDataset,
        testing_ds: PhynnDataset,
        epochs: int,
    ) -> nn.Module:
        optimizer = self._optimizer_cls(model.parameters(), lr=self._lr)

        training_dl = DataLoader(training_ds, self._batch_size)
        testing_dl = DataLoader(testing_ds, batch_size=self._batch_size, shuffle=False)

        with tqdm(total=epochs, desc="Starting...") as progress_bar:
            for epoch in range(epochs):
                self._train_step(model, optimizer, training_dl, progress_bar, epoch)
                self._test_step(model, testing_dl, progress_bar, epoch)

        return model

    def _train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        training_dl: DataLoader,
        progress_bar: tqdm,
        epoch: int,
    ) -> float:
        model.train()

        losses = []

        for i, (X, Y) in enumerate(training_dl):
            loss = self._forward_get_loss(X, Y, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            progress_bar.set_description(
                f"[TRAIN] Epoch: {epoch} Batch: {i+1}/{len(training_dl)} Loss (current): {loss.item()}"
            )
            progress_bar.update()

        return sum(losses) / len(losses)

    def _test_step(
        self,
        model: nn.Module,
        testing_dl: DataLoader,
        progress_bar: tqdm,
        epoch: int,
    ) -> float:
        model.eval()

        losses = []

        with th.no_grad():
            for i, (X, Y) in enumerate(testing_dl):
                loss = self._forward_get_loss(X, Y, model)
                losses.append(loss.item())

                progress_bar.set_description(
                    f"[TEST] Epoch: {epoch} Batch: {i+1}/{len(testing_dl)} Loss (mean): {sum(losses) / len(losses)}"
                )
                progress_bar.update()

        return sum(losses) / len(losses)

    def _forward_get_loss(self, X, Y, model: nn.Module) -> th.Tensor:
        Y_computed = model(*X)
        return self._loss_fun(Y_computed[:2], Y) + Y_computed[2].mean()
