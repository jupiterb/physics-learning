import torch as th

from torch import nn
from typing import Sequence

from phynn.nn.autoencoder.base import AutoEncoder
from phynn.nn.elementary.activation import ActivationFunction
from phynn.nn.elementary.conv import Conv


class ConvAutoEncoder(AutoEncoder):
    def __init__(
        self,
        image_shape: Sequence[int],
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        poolings: Sequence[int] | None = None,
        activation_functions: Sequence[ActivationFunction] | None = None,
        is_3d: bool = False,
        inner: AutoEncoder | None = None,
        flat_latent: bool = True,
    ) -> None:
        encoder_paddings = [(size - 1) // 2 for size in kernel_sizes]

        encoder = Conv(
            channels,
            kernel_sizes,
            None,
            encoder_paddings,
            poolings,
            activation_functions=activation_functions,
            transpose=False,
            is_3d=is_3d,
        )

        decoder_channels = channels[::-1]
        decoder_activations_functions = (
            None if activation_functions is None else activation_functions[::-1]
        )
        decoder_kernel_sizes = kernel_sizes[::-1]
        decoder_poolings = None if poolings is None else poolings[::-1]
        decoder_paddings = [(size - 1) // 2 for size in decoder_kernel_sizes]

        decoder = Conv(
            decoder_channels,
            decoder_kernel_sizes,
            None,
            decoder_paddings,
            decoder_poolings,
            decoder_activations_functions,
            transpose=True,
            is_3d=is_3d,
        )

        if flat_latent:
            original_latent_shape = encoder(th.zeros((1, *image_shape))).shape[1:]
            encoder = nn.Sequential(encoder, nn.Flatten())
            unflatten = nn.Unflatten(dim=1, unflattened_size=original_latent_shape)
            decoder = nn.Sequential(unflatten, decoder)

        super().__init__(encoder, decoder, inner)
