import torch as th

from torch import nn


class ScaledLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, target_scale: float = 1) -> None:
        super(ScaledLoss, self).__init__()
        self._loss = base_loss
        self._target_scale = target_scale

    def forward(self, Y_pred: th.Tensor, Y_target: th.Tensor) -> th.Tensor:
        maxes, _ = Y_target.view(Y_target.size(0), -1).max(dim=1)
        shape = -1, *tuple(1 for _ in Y_target.shape[1:])
        scales = (self._target_scale / maxes).reshape(shape)

        Y_pred = Y_pred * scales
        Y_target = Y_target * scales

        return self._loss(Y_pred, Y_target)
