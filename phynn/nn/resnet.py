import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Generic, Union

from phynn.nn.base import NNInitParams, NNBlockParams, NNBuilder


@dataclass
class ResBlockParams(Generic[NNBlockParams]):
    params: NNBlockParams
    size: int


MaybeResBlockParams = Union[ResBlockParams[NNBlockParams], NNBlockParams]


class ResBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super(ResBlock, self).__init__()
        self._block = block

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._block(x) + x


class ResNet(NNBuilder[NNInitParams, MaybeResBlockParams[NNBlockParams]]):
    def __init__(self, nn_builder: NNBuilder[NNInitParams, NNBlockParams]) -> None:
        self._builder = nn_builder

    def init(
        self, params: NNInitParams
    ) -> NNBuilder[NNInitParams, MaybeResBlockParams[NNBlockParams]]:
        self._builder.init(params)
        self._block_params = []
        return self

    def prepend(
        self, params: MaybeResBlockParams[NNBlockParams]
    ) -> NNBuilder[NNInitParams, MaybeResBlockParams[NNBlockParams]]:
        self._block_params = [params] + self._block_params
        return self

    def append(
        self, params: MaybeResBlockParams[NNBlockParams]
    ) -> NNBuilder[NNInitParams, MaybeResBlockParams[NNBlockParams]]:
        self._block_params.append(params)
        return self

    def reset(
        self, keep_end: bool
    ) -> NNBuilder[NNInitParams, MaybeResBlockParams[NNBlockParams]]:
        self._builder = self._builder.reset(keep_end)
        return self

    def build(self) -> nn.Sequential:
        resnet = nn.Sequential()

        for params in self._block_params:
            resnet.append(self._create_block(params))
            self._builder.reset(keep_end=True)

        return resnet

    def _create_block(self, params: MaybeResBlockParams[NNBlockParams]) -> nn.Module:
        if isinstance(params, ResBlockParams):
            for _ in range(params.size):
                self._builder.append(params.params)
            return ResBlock(self._builder.build())
        else:
            return self._builder.append(params).build()
