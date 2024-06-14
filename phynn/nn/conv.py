import torch.nn as nn

from dataclasses import dataclass
from typing import Type

from phynn.nn.base import NNBuilder


@dataclass
class ConvInitParams:
    initial_channels: int


@dataclass
class ConvBlockParams:
    channels: int
    kernel_size: int = 3
    activation: Type[nn.Module] = nn.LeakyReLU
    batch_norm: bool = True
    stride: int = 1
    same_padding: bool = True
    rescale: int = 1
    dropout: float = 0.0


@dataclass
class _ConvType:
    transpose: bool
    upsample: bool
    rescale_on_begin: bool


@dataclass
class _ActualConvBlockParams:
    in_channels: int
    out_channels: int
    conv_type: _ConvType
    details: ConvBlockParams


def ConvBlock(params: _ActualConvBlockParams) -> nn.Sequential:
    p = params
    d = p.details
    t = p.conv_type

    padding = (d.kernel_size - 1) // 2 if d.same_padding else 1
    conv_cls = nn.ConvTranspose2d if t.transpose else nn.Conv2d

    cnn = nn.Sequential()

    def rescale():
        if t.upsample:
            cnn.append(nn.Upsample(scale_factor=d.rescale, mode="bilinear"))
        else:
            cnn.append(nn.MaxPool2d(kernel_size=d.rescale, stride=d.rescale))

    if d.rescale > 1 and t.rescale_on_begin:
        rescale()

    if d.dropout > 0:
        cnn.append(nn.Dropout(d.dropout))

    conv = conv_cls(p.in_channels, p.out_channels, d.kernel_size, d.stride, padding)
    cnn.append(conv)

    if d.batch_norm:
        cnn.append(nn.BatchNorm2d(p.out_channels))

    cnn.append(d.activation())

    if d.rescale > 1 and not t.rescale_on_begin:
        rescale()

    return cnn


class Conv(NNBuilder[ConvInitParams, ConvBlockParams]):
    def __init__(
        self,
        transpose: bool = False,
        upsample: bool = False,
        rescale_on_begin: bool = False,
    ) -> None:
        self._conv_type = _ConvType(transpose, upsample, rescale_on_begin)

    def init(
        self, params: ConvInitParams
    ) -> NNBuilder[ConvInitParams, ConvBlockParams]:
        self._in_channels = params.initial_channels
        self._out_channels = params.initial_channels
        self._block_params = []
        return self

    def prepend(
        self, params: ConvBlockParams
    ) -> NNBuilder[ConvInitParams, ConvBlockParams]:
        actual_params = _ActualConvBlockParams(
            params.channels, self._in_channels, self._conv_type, params
        )
        self._block_params = [actual_params] + self._block_params
        self._in_channels = params.channels
        return self

    def append(
        self, params: ConvBlockParams
    ) -> NNBuilder[ConvInitParams, ConvBlockParams]:
        actual_params = _ActualConvBlockParams(
            self._out_channels, params.channels, self._conv_type, params
        )
        self._block_params.append(actual_params)
        self._out_channels = params.channels
        return self

    def reset(self, keep_end: bool) -> NNBuilder[ConvInitParams, ConvBlockParams]:
        return (
            self.init(ConvInitParams(self._out_channels))
            if keep_end
            else self.init(ConvInitParams(self._in_channels))
        )

    def build(self) -> nn.Sequential:
        conv = nn.Sequential()

        for params in self._block_params:
            conv.append(ConvBlock(params))

        return conv
