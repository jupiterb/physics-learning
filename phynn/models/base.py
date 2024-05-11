from dataclasses import dataclass

import lightning as L
from torch import optim, utils
from typing import Type


@dataclass
class OptimizerParams:
    optimizer: (
        Type[optim.Adam] | Type[optim.AdamW] | Type[optim.RMSprop] | Type[optim.SGD]
    )
    lr: float


class BaseModel(L.LightningModule):
    def __init__(self, optimizer_params: OptimizerParams) -> None:
        super().__init__()
        self._optimizer_params = optimizer_params

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = self._optimizer_params.optimizer(
            self.parameters(), lr=self._optimizer_params.lr
        )
        return optimizer
