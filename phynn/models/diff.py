from lightning.pytorch.loggers import WandbLogger
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

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


class DiffEquationModel(BaseModel):
    def __init__(
        self,
        neural_diff_eq: DiffEquation,
        optimizer_params: OptimizerParams,
        equation_name: str | None = None,
    ) -> None:
        name = "equation" + ("" if equation_name is None else f"_{equation_name}")
        super().__init__(optimizer_params, name)
        self._diff_eq = neural_diff_eq

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u_input, u_target, params, duration = batch
        u_prediction = simulate(self._diff_eq, u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u_input, u_target, params, duration = batch
        u_prediction = simulate(self._diff_eq, u_input, params, duration)
        loss = self._loss(u_prediction, u_target)
        self.log_dict({"val_loss": loss})

        if batch_idx == 0:
            _try_log_val_visualization(self.logger, u_input, u_target, u_prediction)

        return loss

    def _loss(self, prediction: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.mse_loss(prediction, target)


class ForwardProblemDiffEquationModel(BaseModel):
    def __init__(
        self,
        neural_diff_eq: DiffEquation,
        params: Sequence[float],
        optimizer_params: OptimizerParams,
        pde_residuum_weight: float = 0.5,
        equation_name: str | None = None,
    ) -> None:
        name = "equation_forward" + (
            "" if equation_name is None else f"_{equation_name}"
        )
        super().__init__(optimizer_params, name)

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
        self,
        neural_diff_eq: FrozenDiffEquation,
        optimizer_params: OptimizerParams,
        equation_name: str | None = None,
    ) -> None:
        name = "equation_inverse" + (
            "" if equation_name is None else f"_{equation_name}"
        )
        super().__init__(optimizer_params, name)

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
