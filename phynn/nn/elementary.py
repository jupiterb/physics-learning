from __future__ import annotations

import torch as th

from torch import nn
from typing import Type


class FC(nn.Module):
    def __init__(self, initial_features: int) -> None:
        super(FC, self).__init__()
        self._fc = nn.Sequential()
        self._in_features = initial_features
        self._out_features = initial_features

    def add_input_layer(
        self,
        in_features: int,
        activation: Type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = True,
    ) -> FC:
        layer = FC._layer(in_features, self._in_features, activation, batch_norm)

        self._fc = layer + self._fc
        self._in_features = in_features

        return self

    def add_output_layer(
        self,
        out_features: int,
        activation: Type[nn.Module] = nn.LeakyReLU,
        batch_norm: bool = True,
    ) -> FC:
        layer = FC._layer(self._out_features, out_features, activation, batch_norm)

        self._fc = self._fc + layer
        self._out_features = out_features

        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._fc(x)

    @staticmethod
    def _layer(
        in_features: int,
        out_features: int,
        activation: Type[nn.Module],
        batch_norm: bool,
    ) -> nn.Sequential:
        layer = nn.Sequential()

        layer.append(nn.Linear(in_features, out_features))

        if batch_norm:
            layer.append(nn.BatchNorm1d(out_features))

        layer.append(activation())

        return layer


class Conv(nn.Module):
    def __init__(self, initial_channels: int) -> None:
        super(Conv, self).__init__()
        self._conv = nn.Sequential()
        self._in_channels = initial_channels
        self._out_channels = initial_channels

    def add_input_block(
        self,
        in_channels: int,
        kernel_size: int,
        activation: Type[nn.Module] = nn.LeakyReLU,
        stride: int = 1,
        same_padding: bool = True,
        rescale: int = 1,
        batch_norm: bool = True,
        transpose: bool = False,
    ) -> Conv:
        block = Conv._block(
            in_channels,
            self._in_channels,
            kernel_size,
            activation,
            stride,
            same_padding,
            rescale,
            batch_norm,
            transpose,
        )

        self._conv = block + self._conv
        self._in_channels = in_channels

        return self

    def add_output_block(
        self,
        out_channels: int,
        kernel_size: int,
        activation: Type[nn.Module] = nn.LeakyReLU,
        stride: int = 1,
        same_padding: bool = True,
        rescale: int = 1,
        batch_norm: bool = True,
        transpose: bool = False,
    ) -> Conv:
        block = Conv._block(
            self._out_channels,
            out_channels,
            kernel_size,
            activation,
            stride,
            same_padding,
            rescale,
            batch_norm,
            transpose,
        )

        self._conv = self._conv + block
        self._out_channels = out_channels

        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._conv(x)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Type[nn.Module],
        stride: int,
        same_padding: bool,
        rescale: int,
        batch_norm: bool,
        transpose: bool,
    ) -> nn.Sequential:
        block = nn.Sequential()

        conv_type = nn.ConvTranspose2d if transpose else nn.Conv2d
        padding = (kernel_size - 1) // 2 if same_padding else 1

        if rescale > 1 and transpose:
            block.append(nn.Upsample(scale_factor=rescale, mode="nearest"))

        block.append(conv_type(in_channels, out_channels, kernel_size, stride, padding))

        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))

        block.append(activation())

        if rescale > 1 and not transpose:
            block.append(nn.MaxPool2d(kernel_size=rescale, stride=rescale))

        return block
