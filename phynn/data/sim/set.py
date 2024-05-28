import torch as th
from torch.utils.data import Dataset
from typing import Callable, Sequence

from phynn.data.img.interface import ImagesDataInterface
from phynn.diff import DiffEquation, simulate


SimulationDataSample = tuple[
    th.Tensor,  # simulation start
    th.Tensor,  # simulation result
    th.Tensor,  # params
    th.Tensor,  # duration
]


class DynamicSimulationDataset(Dataset):
    def __init__(
        self,
        initial_conditions: ImagesDataInterface,
        diff_eq: DiffEquation,
        params_provider: Callable[[int], th.Tensor],
        max_simulation_steps: int,
        max_pre_steps: int,
    ) -> None:
        super().__init__()
        self._ics = initial_conditions
        self._diff = diff_eq
        self._params = params_provider
        self._max_sim_steps = max_simulation_steps
        self._max_pre_steps = max_pre_steps

    def __len__(self) -> int:
        return self._ics.size

    def __getitem__(self, index: int) -> SimulationDataSample:
        starts, results, params, sim_steps = self.__getitems__([index])[0]
        return starts[0], results[0], params[0], sim_steps[0]

    def __getitems__(self, indices: Sequence[int]) -> SimulationDataSample:
        batch_size = len(indices)

        ics = self._ics.get_batch(indices)
        params = self._params(len(indices))
        pre_steps = th.randint(low=0, high=self._max_pre_steps, size=(batch_size,))
        sim_steps = th.randint(low=1, high=self._max_sim_steps, size=(batch_size,))

        with th.no_grad():
            starts = simulate(self._diff, ics, params, pre_steps)
            results = simulate(self._diff, starts, params, sim_steps)

        return starts, results, params, sim_steps
