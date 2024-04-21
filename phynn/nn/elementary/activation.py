from torch import nn
from typing import Type


ActivationFunction = (
    Type[nn.ReLU]
    | Type[nn.Tanh]
    | Type[nn.Sigmoid]
    | Type[nn.LeakyReLU]
    | Type[nn.SELU]
)
