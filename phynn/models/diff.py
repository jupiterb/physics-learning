from lightning.pytorch.loggers import WandbLogger
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from dataclasses import dataclass
from typing import Sequence

from phynn.models.base import BaseModel, OptimizerParams
from phynn.diff import DiffEquation, FrozenDiffEquation, simulate


def _try_log_val_visualization(
    logger,
    u_input: th.Tensor,
    u_target: th.Tensor,
    u_prediction: th.Tensor,
) -> None:
    if isinstance(logger, WandbLogger):
        u_input_grid = make_grid(u_input, 1)
        u_target_grid = make_grid(u_target, 1)
        u_prediction_gird = make_grid(u_prediction, 1)

        logger.log_image(
            key="val_visualization",
            images=[u_input_grid, u_target_grid, u_prediction_gird],
            caption=["u_input", "u_target", "u_prediction"],
        )


@dataclass
class _DiffLossInfo:
    loss: th.Tensor
    mass_loss: th.Tensor
    image_loss: th.Tensor

    def asdict(self) -> dict[str, th.Tensor]:
        return {
            "loss": self.loss,
            "mass_loss": self.mass_loss,
            "image_loss": self.image_loss,
        }


class _DiffEquationLoss:
    def __init__(self, image_shape: Sequence[int]) -> None:
        self._mass_loss_weight = 1 / th.prod(th.Tensor(image_shape))
        self._mass_dims = list(range(1, len(image_shape)))

    def __call__(self, prediction: th.Tensor, target: th.Tensor) -> _DiffLossInfo:
        image_loss = F.mse_loss(prediction, target)

        prediction_mass = prediction.sum(self._mass_dims)
        target_mass = target.sum(self._mass_dims)

        mass_loss = F.mse_loss(prediction_mass, target_mass) * self._mass_loss_weight

        loss = image_loss + mass_loss

        return _DiffLossInfo(loss, mass_loss, image_loss)


class DiffEquationModel(BaseModel):
    def __init__(
        self,
        neural_diff_eq: DiffEquation,
        input_shape: Sequence[int],
        optimizer_params: OptimizerParams,
    ) -> None:
        super().__init__(optimizer_params)
        self._diff_eq = neural_diff_eq
        self._loss_fun = _DiffEquationLoss(input_shape)

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u_input, u_target, params, duration = batch
        u_prediction = simulate(self._diff_eq, u_input, params, duration)
        loss_info = self._loss_fun(u_prediction, u_target)
        self.log_dict(loss_info.asdict())
        return loss_info.loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u_input, u_target, params, duration = batch
        u_prediction = simulate(self._diff_eq, u_input, params, duration)
        loss_info = self._loss_fun(u_prediction, u_target)
        self.log_dict(
            {"val_" + name: value for name, value in loss_info.asdict().items()}
        )

        if batch_idx == 0:
            _try_log_val_visualization(self.logger, u_input, u_target, u_prediction)

        return loss_info.loss


class ForwardProblemDiffEquationModel(BaseModel):
    def __init__(
        self,
        neural_diff_eq: DiffEquation,
        params: Sequence[float],
        optimizer_params: OptimizerParams,
        pde_residuum_weight: float = 0.5,
    ) -> None:
        super().__init__(optimizer_params)

        self._diff_eq = neural_diff_eq

        self._params = th.Tensor(params)
        self._num_params = len(params)

        self._pde_weight = pde_residuum_weight

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_result_d, u_result_p, duration = batch[0], batch[1], batch[2], batch[3]
        loss = self._step(u, u_result_d, u_result_p, duration)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_result_d, u_result_p, duration = batch[0], batch[1], batch[2], batch[3]
        loss = self._step(u, u_result_d, u_result_p, duration)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(
        self,
        u: th.Tensor,
        u_result_d: th.Tensor,
        u_result_p: th.Tensor,
        duration: th.Tensor,
    ) -> th.Tensor:
        params = self._params.view((1, self._num_params))
        u_result = simulate(self._diff_eq, u, params, duration)
        loss_data = F.mse_loss(u_result - u, u_result_d - u)
        loss_pde = F.mse_loss(u_result - u, u_result_p - u)
        return (1 - self._pde_weight) * loss_data + self._pde_weight * loss_pde


class InverseProblemDiffEquationModel(BaseModel):
    def __init__(
        self, neural_diff_eq: FrozenDiffEquation, optimizer_params: OptimizerParams
    ) -> None:
        super().__init__(optimizer_params)

        self._diff_eq = neural_diff_eq

        self._num_params = neural_diff_eq.num_components
        self._params = nn.Parameter(th.empty((self._num_params,)))

    @property
    def params(self) -> th.Tensor:
        return self._params.detach()

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_result_d, duration = batch[0], batch[1], batch[2]
        loss = self._step(u, u_result_d, duration)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_result_d, duration = batch[0], batch[1], batch[2]
        loss = self._step(u, u_result_d, duration)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(
        self, u: th.Tensor, u_result: th.Tensor, duration: th.Tensor
    ) -> th.Tensor:
        params = self._params.view((1, self._num_params))
        u_result_predicted = simulate(self._diff_eq, u, params, duration)
        return F.mse_loss(u_result_predicted - u, u_result - u)
