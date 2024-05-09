from __future__ import annotations

from torch import Tensor, zeros, nn
from typing import Generic, Sequence

from phynn.nn.base import NNBlockParams, NNBuilder


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_shape: Sequence[int],
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super(AutoEncoder, self).__init__()
        self._in_shape = in_shape
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

    def flatten(self) -> AutoEncoder:
        if len(self._in_shape) == 1:
            return self

        latent_shape = self._encoder(zeros((1, *self._in_shape))).shape[1:]

        self._encoder = nn.Sequential(self._encoder, nn.Flatten())
        self._decoder = nn.Sequential(nn.Unflatten(1, latent_shape), self._decoder)

        return self

    def forward(self, x: Tensor) -> Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)


class AutoEncoderBuilder(AutoEncoder, Generic[NNBlockParams]):
    def __init__(
        self,
        in_shape: Sequence[int],
        encoder_builder: NNBuilder[NNBlockParams],
        decoder_builder: NNBuilder[NNBlockParams],
    ) -> None:
        super().__init__(in_shape, encoder_builder.nn, decoder_builder.nn)
        self._encoder_builder = encoder_builder
        self._decoder_builder = decoder_builder

    def add_block(self, params: NNBlockParams) -> AutoEncoderBuilder:
        self._encoder_builder.append(params)
        self._decoder_builder.prepend(params)
        return self
