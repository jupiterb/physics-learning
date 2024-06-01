import torch as th
import torch.nn as nn
from typing import Sequence


class DiffEquation(nn.Module):
    def __init__(self, neural_eq_components: Sequence[nn.Module]) -> None:
        super(DiffEquation, self).__init__()
        self._nns = nn.ModuleList(neural_eq_components)

    @property
    def num_components(self) -> int:
        return len(self._nns)

    def _components(self, u: th.Tensor) -> th.Tensor:
        return th.cat([nn(u) for nn in self._nns], 1)

    def forward(self, u: th.Tensor, params: th.Tensor) -> th.Tensor:
        components = self._components(u)
        expand_dims = params.shape + (1,) * (u.ndim - 2)
        return (components * params.view(expand_dims)).sum(1, keepdim=True)


class FrozenDiffEquation(DiffEquation):
    def __init__(self, neural_eq_components: Sequence[nn.Module]) -> None:
        super().__init__(neural_eq_components)

    def _components(self, u: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return super()._components(u)


def simulate(
    diff_eq: DiffEquation,
    initial_conditions: th.Tensor,
    params: th.Tensor,
    duration: th.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> th.Tensor:
    max_duration = int(duration.max().item())
    u = initial_conditions.clone()

    for time in range(max_duration):
        time_mask = duration > time
        time_mask = time_mask.squeeze() if time_mask.dim() > 1 else time_mask

        diff = diff_eq(u[time_mask], params[time_mask])

        u[time_mask] += diff
        u = u.clip(min_value, max_value)

    return u
