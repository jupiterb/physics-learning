from __future__ import annotations

import torch as th
import torch.nn as nn

from dataclasses import dataclass
from typing import Generic

from phynn.nn.base import NNInitParams, NNBlockParams, NNBuilder


class UNetLevel(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, sublevel: nn.Module
    ) -> None:
        super(UNetLevel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._sublevel = sublevel

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_enc = self._encoder(x)
        x_sub = self._sublevel(x_enc)
        x_cat = th.concat([x_enc, x_sub], dim=1)
        return self._decoder(x_cat)


@dataclass
class _UNetLevelParams(Generic[NNBlockParams]):
    encoder_params: NNBlockParams
    decoder_params: NNBlockParams


class UNet(Generic[NNInitParams, NNBlockParams]):
    def __init__(
        self,
        encoder_builder: NNBuilder[NNInitParams, NNBlockParams],
        decoder_builder: NNBuilder[NNInitParams, NNBlockParams],
    ) -> None:
        super().__init__()
        self._encoder_builder = encoder_builder
        self._decoder_builder = decoder_builder

    def init(self, params: NNInitParams) -> UNet[NNInitParams, NNBlockParams]:
        self._encoder_builder.init(params)
        self._decoder_builder.init(params)
        self._level_params: list[list[_UNetLevelParams[NNBlockParams]]] = [[]]
        return self

    def add_symmetrical_blocks(
        self, params: NNBlockParams
    ) -> UNet[NNInitParams, NNBlockParams]:
        self._level_params[-1].append(_UNetLevelParams(params, params))
        return self

    def split_level(
        self, encoder_params: NNBlockParams, concat_decoder_params: NNBlockParams
    ) -> UNet[NNInitParams, NNBlockParams]:
        level_params = _UNetLevelParams(encoder_params, concat_decoder_params)
        self._level_params[-1].append(level_params)
        self._level_params.append([])
        return self

    def build(self) -> nn.Module:
        level_params = self._level_params.pop(0)

        if len(self._level_params) == 0:
            return self._build_bridge(level_params)
        else:
            return self._build_unet_level(level_params)

    def _build_bridge(self, level_params: list[_UNetLevelParams]) -> nn.Module:
        for params in level_params:
            self._encoder_builder.append(params.encoder_params)
            self._decoder_builder.prepend(params.decoder_params)

        encoder = self._encoder_builder.build()
        decoder = self._decoder_builder.build()

        return encoder + decoder

    def _build_unet_level(self, level_params: list[_UNetLevelParams]) -> nn.Module:
        for params in level_params[:-1]:
            self._encoder_builder.append(params.encoder_params)
            self._decoder_builder.prepend(params.decoder_params)

        self._encoder_builder.append(level_params[-1].encoder_params)
        encoder = self._encoder_builder.build()

        decoder_after_concat = self._decoder_builder.build()
        self._decoder_builder.reset(False)

        self._decoder_builder.prepend(level_params[-1].decoder_params)
        concat_decoder = self._decoder_builder.build()

        decoder = concat_decoder + decoder_after_concat

        self._encoder_builder.reset(True)
        self._decoder_builder.reset(True)

        sublevel = self.build()

        return UNetLevel(encoder, decoder, sublevel)
