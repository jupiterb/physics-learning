import torch as th

from torch import nn

from phynn.nn import AutoEncoder


class PhysicsInformedAutoEncoder(nn.Module):
    def __init__(
        self, autoencoder: AutoEncoder, delta_latent_encoder: nn.Module
    ) -> None:
        super(PhysicsInformedAutoEncoder, self).__init__()
        self._autoencoder = autoencoder
        self._delta_latent_encoder = delta_latent_encoder

    def forward(
        self, x: th.Tensor, y: th.Tensor, t: th.Tensor, params: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_x = self._autoencoder.encoder(x)
        latent_y = self._autoencoder.encoder(y)

        t_params = th.hstack((t, params))
        delta_latent = self._delta_latent_encoder(t_params)

        delta_latent_error = delta_latent - (latent_y - latent_x)

        return (
            self._autoencoder.decoder(latent_x),
            self._autoencoder.decoder(latent_y),
            delta_latent_error,
        )
