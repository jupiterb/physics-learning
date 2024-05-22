import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from phynn.models.base import BaseModel, OptimizerParams
from phynn.nn import DiffEquation, FrozenDiffEquation


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
        u, u_next, params = batch[0], batch[1], batch[2]
        loss = self._step(u, u_next, params)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_next, params = batch[0], batch[1], batch[2]
        loss = self._step(u, u_next, params)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(self, u: th.Tensor, u_next: th.Tensor, params: th.Tensor) -> th.Tensor:
        diff = self._diff_eq(u, params)
        return F.mse_loss(diff, u_next - u)


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
        u, u_next_data, u_next_pde = batch[0], batch[1], batch[2]
        loss = self._step(u, u_next_data, u_next_pde)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_next_data, u_next_pde = batch[0], batch[1], batch[2]
        loss = self._step(u, u_next_data, u_next_pde)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(
        self, u: th.Tensor, u_next_data: th.Tensor, u_next_pde: th.Tensor
    ) -> th.Tensor:
        params = self._params.view((1, self._num_params))
        diff = self._diff_eq(u, params)
        loss_data = F.mse_loss(diff, u_next_data - u)
        loss_pde = F.mse_loss(diff, u_next_pde - u)
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
        u, u_next = batch[0], batch[1]
        loss = self._step(u, u_next)
        self.log_dict({"loss": loss})
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:  # type: ignore
        u, u_next = batch[0], batch[1]
        loss = self._step(u, u_next)
        self.log_dict({"val_loss": loss})
        return loss

    def _step(self, u: th.Tensor, u_next: th.Tensor) -> th.Tensor:
        params = self._params.view((1, self._num_params))
        diff = self._diff_eq(u, params)
        return F.mse_loss(diff, u_next - u)
