import torch as th

from torch import nn
from typing import Sequence

from phynn.nn.elementary.activation import ActivationFunction


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pooling: int | None,
        activation_fun,
        transpose: bool,
        is_3d: bool,
    ) -> None:
        nn.Module.__init__(self)

        if transpose:
            conv_fun = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        else:
            conv_fun = nn.Conv3d if is_3d else nn.Conv2d

        self._conv = nn.Sequential()
        self._conv.append(
            conv_fun(in_channels, out_channels, kernel_size, stride, padding)
        )
        self._conv.append(activation_fun())

        if pooling is not None:
            if transpose and is_3d:
                pool = nn.Upsample(scale_factor=pooling, mode="nearest")
            elif is_3d:
                pool = nn.MaxPool3d(kernel_size=pooling, stride=pooling)
            else:
                pool = nn.MaxPool2d(kernel_size=pooling, stride=pooling)

            self._conv.append(pool)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._conv(x)


class Conv(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int] | None = None,
        paddings: Sequence[int] | None = None,
        poolings: Sequence[int | None] | None = None,
        activation_functions: Sequence[ActivationFunction] | None = None,
        transpose: bool = False,
        is_3d: bool = False,
    ) -> None:
        super(Conv, self).__init__()

        self._conv = nn.Sequential()

        sequence_of = lambda x: [x for _ in kernel_sizes]

        strides = sequence_of(1) if strides is None else strides
        paddings = sequence_of(1) if paddings is None else paddings
        poolings = sequence_of(None) if poolings is None else poolings
        activation_functions = (
            sequence_of(nn.ReLU)
            if activation_functions is None
            else activation_functions
        )

        in_channels = channels[0]

        for out_channels, kernel_size, stride, padding, pooling, activation_fun in zip(
            channels[1:],
            kernel_sizes,
            strides,
            paddings,
            poolings,
            activation_functions,
        ):
            block = _ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                pooling,
                activation_fun,
                transpose,
                is_3d,
            )
            self._conv.append(block)
            in_channels = out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._conv(x)
