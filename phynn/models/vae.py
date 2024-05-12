from dataclasses import dataclass, asdict
import torch as th
import torch.nn.functional as F

from phynn.models.base import BaseModel, OptimizerParams
from phynn.nn import VariationalAutoEncoder


@dataclass
class _VAEStepInfo:
    loss: th.Tensor
    reconstruction_loss: th.Tensor
    kld_loss: th.Tensor


class VAEModel(BaseModel):
    def __init__(
        self, vae: VariationalAutoEncoder, optimizer_params: OptimizerParams
    ) -> None:
        super().__init__(optimizer_params, "vae")
        self._vae = vae
        self._kld_weight = 1 / th.prod(th.Tensor(vae.in_shape))

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        x = batch[0]
        info = self._step(x)
        self.log_dict(asdict(info))
        return info.loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        x = batch[0]
        info = self._step(x)
        self.log_dict({"val_" + name: value for name, value in asdict(info).items()})
        return info.loss

    def _step(self, x: th.Tensor) -> _VAEStepInfo:
        mu, log_var = self._vae.encoder(x)
        x_hat = self._vae.decoder(mu, log_var)

        reconstruction_loss = self._reconstruction_loss(x, x_hat)
        kld_loss = self._kld_loss(mu, log_var)
        loss = reconstruction_loss + kld_loss

        return _VAEStepInfo(loss, reconstruction_loss, kld_loss)

    def _reconstruction_loss(self, x: th.Tensor, x_hat: th.Tensor) -> th.Tensor:
        return F.mse_loss(x_hat, x)

    def _kld_loss(self, mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        var = th.exp(log_var)
        return (
            -0.5
            * self._kld_weight
            * th.mean(th.sum(1 + log_var - var - mu**2, dim=1), dim=0)
        )
