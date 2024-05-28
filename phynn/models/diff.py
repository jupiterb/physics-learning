import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from phynn.models.base import BaseModel, OptimizerParams
from phynn.diff import DiffEquation, FrozenDiffEquation, simulate


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
        u, u_result, params, duration = batch[0], batch[1], batch[2], batch[3]
        loss = self._step(u, u_result, params, duration)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_result, params, duration = batch[0], batch[1], batch[2], batch[3]
        loss = self._step(u, u_result, params, duration)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(
        self, u: th.Tensor, u_result: th.Tensor, params: th.Tensor, duration: th.Tensor
    ) -> th.Tensor:
        u_result_predicted = simulate(self._diff_eq, u, params, duration)
        return F.mse_loss(u_result_predicted - u, u_result - u)


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
