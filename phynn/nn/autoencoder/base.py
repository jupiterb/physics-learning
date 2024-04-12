from __future__ import annotations

import torch as th

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        inner: AutoEncoder | None,
    ) -> None:
        super(AutoEncoder, self).__init__()

        if inner is not None:
            encoder = nn.Sequential(encoder, inner.encoder)
            decoder = nn.Sequential(inner.decoder, decoder)

        self._encoder = encoder
        self._decoder = decoder

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder

    def forward(self, x: th.Tensor) -> th.Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)
