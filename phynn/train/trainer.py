import torch as th

from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Type
from tqdm import tqdm

from phynn.train.dataset import PhynnDataset


OptimizerClass = Type[optim.Adam] | Type[optim.AdamW] | Type[optim.SGD]


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

        with tqdm(total=epochs, desc="first epoch running...") as progress_bar:
            for epoch in range(epochs):
                train_loss = self._train_step(model, optimizer, training_dl)
                test_loss = self._test_step(model, testing_dl)

                progress_bar.set_description(
                    f"epochs: {epoch+1}  train loss: {train_loss}  test Loss: {test_loss}"
                )
                progress_bar.update()

        return model

    def _train_step(
        self, model: nn.Module, optimizer: optim.Optimizer, training_dl: DataLoader
    ) -> float:
        model.train()

        losses = []

        for X, Y in training_dl:
            loss = self._forward_get_loss(X, Y, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses)

    def _test_step(self, model: nn.Module, testing_dl: DataLoader) -> float:
        model.eval()

        losses = []

        with th.no_grad():
            for X, Y in testing_dl:
                loss = self._forward_get_loss(X, Y, model)
                losses.append(loss.item())

        return sum(losses) / len(losses)

    def _forward_get_loss(self, X, Y, model: nn.Module) -> th.Tensor:
        Y_computed = model(*X)
        return self._loss_fun(Y_computed, Y)
