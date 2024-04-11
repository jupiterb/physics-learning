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
    def __init__(self, *params: float) -> None:
        super().__init__()
        self._constants = tuple(params)

    def _params(self, x: th.Tensor) -> th.Tensor:
        return th.tensor(self._constants)


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
        mask_channel: int,
        min_value: float = 0,
        max_value: float = 1,
    ) -> None:
        super(PDEEval, self).__init__()

        self._pde = pde
        self._params_provider = params_provider
        self._mask_channel = mask_channel

        self._min_value = min_value
        self._max_value = max_value

    def forward(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        steps = int(t.max().item())

        params = self._params_provider(x)
        y = x[:, self._mask_channel].clone().unsqueeze(1)

        for i in range(steps):
            t_mask = t > i
            t_mask = t_mask.squeeze() if t_mask.dim() > 1 else t_mask

            diff = self._pde(y[t_mask], params[t_mask]).requires_grad_(True)

            y[t_mask] += diff

            y = y.clip(self._min_value, self._max_value)

        return y
