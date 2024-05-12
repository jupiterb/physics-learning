from dataclasses import dataclass

import lightning as L
from torch import optim
from typing import Type


@dataclass
class OptimizerParams:
    optimizer: (
        Type[optim.Adam] | Type[optim.AdamW] | Type[optim.RMSprop] | Type[optim.SGD]
    )
    lr: float


class BaseModel(L.LightningModule):
    def __init__(self, optimizer_params: OptimizerParams, name: str) -> None:
        super().__init__()
        self._optimizer_params = optimizer_params
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = self._optimizer_params.optimizer(
            self.parameters(), lr=self._optimizer_params.lr
        )
        return optimizer
