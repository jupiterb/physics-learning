import torch as th

from torch import nn

from phynn.nn.autoencoder.base import AutoEncoder


class PhysicsInformedAutoEncoder(nn.Module):
    def __init__(
        self, autoencoder: AutoEncoder, delta_latent_encoder: nn.Module
    ) -> None:
        super(PhysicsInformedAutoEncoder, self).__init__()
        self._autoencoder = autoencoder
        self._delta_latent_encoder = delta_latent_encoder

    def forward(
        self, x: th.Tensor, t: th.Tensor, params: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        latent_x = self._autoencoder.encoder(x)

        t_params = th.hstack((t, params))
        delta_latent = self._delta_latent_encoder(t_params)

        latent_y = latent_x + delta_latent

        return (
            self._autoencoder.decoder(latent_x),
            self._autoencoder.decoder(latent_y),
        )


class DeltaLatentAutoEncoder(nn.Module):
    def __init__(
        self, delta_latent_encoder: nn.Module, delta_latent_decoder: nn.Module
    ) -> None:
        super(DeltaLatentAutoEncoder, self).__init__()
        self._delta_latent_encoder = delta_latent_encoder
        self._delta_latent_decoder = delta_latent_decoder

    def forward(self, t: th.Tensor, params: th.Tensor) -> th.Tensor:
        t_params = th.hstack((t, params))

        with th.no_grad():
            delta_latent = self._delta_latent_encoder(t_params)

        return self._delta_latent_decoder(delta_latent)
