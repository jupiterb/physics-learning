import torch as th
import torch.nn as nn
from typing import Sequence


class DiffEquation(nn.Module):
    def __init__(self, neural_eq_components: Sequence[nn.Module]) -> None:
        super(DiffEquation, self).__init__()
        self._nns = neural_eq_components

    @property
    def num_components(self) -> int:
        return len(self._nns)

    def _components(self, u: th.Tensor) -> th.Tensor:
        return th.stack([nn(u) for nn in self._nns], 1)

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        components = self._components(u)
        expand_dims = params.shape + (1,) * (u.ndim - 2)
        return (components * params.view(expand_dims)).sum(1)


class FrozenDiffEquation(DiffEquation):
    def __init__(self, neural_eq_components: Sequence[nn.Module]) -> None:
        super().__init__(neural_eq_components)

    def _components(self, u: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return super()._components(u)
