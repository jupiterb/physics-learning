import torch.nn as nn

from dataclasses import dataclass
from typing import Type

from phynn.nn.base import NNBuilder


@dataclass
class FCInitParams:
    initial_features: int


@dataclass
class FCBlockParams:
    features: int
    activation: Type[nn.Module] = nn.LeakyReLU


@dataclass
class _ActualFCBlockParams:
    in_features: int
    out_features: int
    activation: Type[nn.Module] = nn.LeakyReLU


def FCBlock(params: _ActualFCBlockParams) -> nn.Sequential:
    fc = nn.Sequential()
    fc.append(nn.Linear(params.in_features, params.out_features))
    fc.append(params.activation())
    return fc


class FC(NNBuilder[FCInitParams, FCBlockParams]):
    def init(self, params: FCInitParams) -> NNBuilder[FCInitParams, FCBlockParams]:
        self._in_features = params.initial_features
        self._out_features = params.initial_features
        self._block_params = []
        return self

    def prepend(self, params: FCBlockParams) -> NNBuilder[FCInitParams, FCBlockParams]:
        actual_params = _ActualFCBlockParams(
            params.features, self._in_features, params.activation
        )
        self._block_params = [actual_params] + self._block_params
        self._in_features = params.features
        return self

    def append(self, params: FCBlockParams) -> NNBuilder[FCInitParams, FCBlockParams]:
        actual_params = _ActualFCBlockParams(
            self._out_features, params.features, params.activation
        )
        self._block_params.append(actual_params)
        self._out_features = params.features
        return self

    def reset(self, keep_end: bool) -> NNBuilder[FCInitParams, FCBlockParams]:
        return (
            self.init(FCInitParams(self._out_features))
            if keep_end
            else self.init(FCInitParams(self._in_features))
        )

    def build(self) -> nn.Sequential:
        fc = nn.Sequential()

        for params in self._block_params:
            fc.append(FCBlock(params))

        return fc
