from __future__ import annotations

from dataclasses import dataclass
from torch import nn
from typing import Type

from phynn.nn.base import NNBuilder


@dataclass
class ConvBlockParams:
    channels: int
    kernel_size: int
    activation: Type[nn.Module] = nn.LeakyReLU
    batch_norm: bool = True
    stride: int = 1
    same_padding: bool = True
    rescale: int = 1


def ConvBlock(
    in_channels: int,
    out_channels: int,
    params: ConvBlockParams,
    transpose: bool = False,
) -> nn.Sequential:
    conv = nn.Sequential()

    if params.rescale > 1 and transpose:
        conv.append(nn.Upsample(scale_factor=params.rescale, mode="nearest"))

    padding = (params.kernel_size - 1) // 2 if params.same_padding else 1
    conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d

    conv.append(
        conv_cls(in_channels, out_channels, params.kernel_size, params.stride, padding)
    )

    if params.batch_norm:
        conv.append(nn.BatchNorm2d(out_channels))

    conv.append(params.activation())

    if params.rescale > 1 and not transpose:
        conv.append(nn.MaxPool2d(kernel_size=params.rescale, stride=params.rescale))

    return conv


class Conv(NNBuilder[ConvBlockParams]):
    def __init__(self, initial_channels: int, transpose: bool = False) -> None:
        super().__init__()
        self._in_channels = initial_channels
        self._out_channels = initial_channels
        self._transpose = transpose

    def prepend(self, params: ConvBlockParams) -> NNBuilder[ConvBlockParams]:
        block = ConvBlock(params.channels, self._in_channels, params, self._transpose)
        self._nn = block + self._nn
        self._in_channels = params.channels
        return self

    def append(self, params: ConvBlockParams) -> NNBuilder[ConvBlockParams]:
        block = ConvBlock(self._out_channels, params.channels, params, self._transpose)
        self._nn = self._nn + block
        self._out_channels = params.channels
        return self
