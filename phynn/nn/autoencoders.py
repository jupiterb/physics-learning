from __future__ import annotations

import torch as th

from torch import nn
from typing import Sequence, Type

from phynn.nn.elementary import FC, Conv


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super(AutoEncoder, self).__init__()

        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def add_inner(self, inner: AutoEncoder) -> AutoEncoder:
        self._encoder = nn.Sequential(self._encoder, inner.encoder)
        self._decoder = nn.Sequential(inner.decoder, self._decoder)
        return self

    def forward(self, x: th.Tensor) -> th.Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)


class FCAutoEncoder(AutoEncoder):
    def __init__(self, in_out_features: int) -> None:
        fc_encoder = FC(in_out_features)
        fc_decoder = FC(in_out_features)

        super().__init__(fc_encoder, fc_decoder)

        self._fc_encoder = fc_encoder
        self._fc_decoder = fc_decoder

    def add_hidden_space(
        self,
        hidden_features: int,
        encoder_activation: Type[nn.Module] = nn.LeakyReLU,
        decoder_activation: Type[nn.Module] = nn.LeakyReLU,
    ) -> FCAutoEncoder:
        self._fc_encoder.add_output_layer(hidden_features, encoder_activation)
        self._fc_decoder.add_input_layer(hidden_features, decoder_activation)
        return self


class ConvAutoEncoder(AutoEncoder):
    def __init__(self, in_out_shape: Sequence[int]) -> None:
        conv_encoder = Conv(in_out_shape[0])
        conv_decoder = Conv(in_out_shape[0])

        super().__init__(conv_encoder, conv_decoder)

        self._conv_encoder = conv_encoder
        self._conv_decoder = conv_decoder

        self._in_out_shape = in_out_shape

    def add_hidden_space(
        self,
        hidden_channels: int,
        kernel_size: int,
        encoder_activation: Type[nn.Module] = nn.LeakyReLU,
        decoder_activation: Type[nn.Module] = nn.LeakyReLU,
        rescale: int = 1,
    ) -> ConvAutoEncoder:
        self._conv_encoder.add_output_block(
            hidden_channels, kernel_size, encoder_activation, rescale=rescale
        )
        self._conv_decoder.add_input_block(
            hidden_channels,
            kernel_size,
            decoder_activation,
            rescale=rescale,
            transpose=True,
        )
        return self

    def flatten(self) -> ConvAutoEncoder:
        latent_shape = self._encoder(th.zeros((1, *self._in_out_shape))).shape[1:]

        self._encoder = nn.Sequential(self._encoder, nn.Flatten())
        self._decoder = nn.Sequential(nn.Unflatten(1, latent_shape), self._decoder)

        return self
