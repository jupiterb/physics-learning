import torch as th

from torch import nn


class Multiply(nn.Module):
    def __init__(self, module: nn.Module, scalar: float) -> None:
        super(Multiply, self).__init__()
        self._module = module
        self._scalar = scalar

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._module(x) * self._scalar


class Froze(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(Froze, self).__init__()
        self._module = module

    def forward(self, x: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return self._module(x)
