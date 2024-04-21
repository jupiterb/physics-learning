import numpy as np
import torch as th

from abc import ABC, abstractmethod
from torch import nn


class PDE(nn.Module, ABC):
    @abstractmethod
    def _diff(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def _np(tensor: th.Tensor) -> np.ndarray:
        return tensor.cpu().detach().numpy()

    def forward(self, x: th.Tensor, params: th.Tensor) -> th.Tensor:
        device = x.device
        diff = self._diff(PDE._np(x), PDE._np(params))
        return th.from_numpy(diff).to(device)


class PDEParamsProvider(nn.Module, ABC):
    def __init__(self) -> None:
        super(PDEParamsProvider, self).__init__()

    @abstractmethod
    def _params(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._params(x)


class PDEStaticParams(PDEParamsProvider):
    def __init__(self, *values: float) -> None:
        super().__init__()
        self._values = tuple(values)

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    @values.setter
    def values(self, *params: float) -> None:
        self._values = tuple(params)

    def _params(self, x: th.Tensor) -> th.Tensor:
        return th.tensor(data=self._values).repeat(len(x), 1).to(x.device)


class PDEDynamicParams(PDEParamsProvider):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model

    def _params(self, x: th.Tensor) -> th.Tensor:
        return self._model(x)


class PDEEval(nn.Module):
    def __init__(
        self,
        pde: PDE,
        params_provider: PDEParamsProvider,
        min_value: float = 0,
        max_value: float = 1,
        min_concentration: float = 0,
    ) -> None:
        super(PDEEval, self).__init__()

        self._pde = pde
        self._params_provider = params_provider

        self._min_value = min_value
        self._max_value = max_value

        self._min_concentration = min_concentration

    def forward(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        steps = int(t.max().item())
        y = x.clone()

        for i in range(steps):
            t_mask = t > i
            t_mask = t_mask.squeeze() if t_mask.dim() > 1 else t_mask

            concentration_mask = y[t_mask] > self._min_concentration
            concentration = y[t_mask] * concentration_mask

            params = self._params_provider(concentration)
            diff = self._pde(concentration, params).requires_grad_(True)

            y[t_mask] += diff

            y = y.clip(self._min_value, self._max_value)

        return y
