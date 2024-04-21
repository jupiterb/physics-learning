import torch as th

from functools import reduce
from torch import nn

from phynn.dataloader import Tensors


class PairwiseLoss(nn.Module):
    def __init__(self, base_loss: nn.Module) -> None:
        super(PairwiseLoss, self).__init__()
        self._loss = base_loss

    def forward(self, Y_preds: Tensors, Y_targets: Tensors) -> th.Tensor:
        losses = [
            self._loss(y_pred, y_target) for y_pred, y_target in zip(Y_preds, Y_targets)
        ]

        return reduce(lambda l0, l1: l0 + l1, losses)
