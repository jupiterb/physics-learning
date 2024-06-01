from __future__ import annotations

from dataclasses import dataclass
import torch as th
import torch.nn as nn
from typing import Callable, Generic

from phynn.nn.base import NNBlockParams, NNBuilder


@dataclass
class ResBlockParams(Generic[NNBlockParams]):
    params: NNBlockParams
    size: int


class ResBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super(ResBlock, self).__init__()
        self._block = block

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._block(x) + x


MaybeResBlockParams = ResBlockParams[NNBlockParams] | NNBlockParams


class ResNet(NNBuilder[MaybeResBlockParams[NNBlockParams]]):
    def __init__(self, nn_builder: NNBuilder[NNBlockParams]) -> None:
        super().__init__()
        self._builder = nn_builder

    def prepend(
        self, params: MaybeResBlockParams[NNBlockParams]
    ) -> NNBuilder[MaybeResBlockParams[NNBlockParams]]:
        block = self._create_block(self._builder.prepend, params)
        self._nn = nn.Sequential(block) + self._nn
        return self

    def append(
        self, params: MaybeResBlockParams[NNBlockParams]
    ) -> NNBuilder[MaybeResBlockParams[NNBlockParams]]:
        block = self._create_block(self._builder.append, params)
        self._nn.append(block)
        return self

    def _create_block(
        self,
        add: Callable[[NNBlockParams], NNBuilder[NNBlockParams]],
        params: MaybeResBlockParams[NNBlockParams],
    ) -> nn.Module:
        if isinstance(params, ResBlockParams):
            for _ in range(params.size):
                add(params.params)
            return ResBlock(self._builder.unload())
        else:
            return add(params).unload()
