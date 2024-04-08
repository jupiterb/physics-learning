import torch as th

from torch import nn
from typing import Sequence

from phynn.nn.elementary.activation import ActivationFunction


class FC(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_functions: Sequence[ActivationFunction] | None = None,
    ) -> None:
        super(FC, self).__init__()

        if activation_functions is None:
            activation_functions = [nn.ReLU for _ in layer_sizes[1:]]

        self._fc = nn.Sequential()
        input_size = layer_sizes[0]

        for output_size, activation_fun in zip(layer_sizes[1:], activation_functions):
            self._fc.append(nn.Linear(input_size, output_size))
            self._fc.append(activation_fun())
            input_size = output_size

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._fc(x)
