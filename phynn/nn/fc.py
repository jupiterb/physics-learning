from __future__ import annotations

from dataclasses import dataclass
from torch import nn
from typing import Type

from phynn.nn.base import NNBuilder


@dataclass
class FCBlockParams:
    features: int
    activation: Type[nn.Module] = nn.LeakyReLU


def FCBlock(
    in_features: int, out_features: int, params: FCBlockParams
) -> nn.Sequential:
    fc = nn.Sequential()
    fc.append(nn.Linear(in_features, out_features))
    fc.append(params.activation())
    return fc


class FC(NNBuilder[FCBlockParams]):
    def __init__(self, initial_features: int) -> None:
        super().__init__()
        self._in_features = initial_features
        self._out_features = initial_features

    def prepend(self, params: FCBlockParams) -> NNBuilder[FCBlockParams]:
        block = FCBlock(params.features, self._in_features, params)
        self._nn = block + self._nn
        self._in_features = params.features
        return self

    def append(self, params: FCBlockParams) -> NNBuilder[FCBlockParams]:
        block = FCBlock(self._out_features, params.features, params)
        self._nn = self._nn + block
        self._out_features = params.features
        return self
