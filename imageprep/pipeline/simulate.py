import numpy as np

from typing import Callable, Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class Simulate(Pipeline):
    def __init__(
        self,
        pipeline: Pipeline,
        simulate: Callable[[np.ndarray, int, tuple[float, ...]], np.ndarray],
        time_distribution: Callable[[], int],
        params_distribution: Callable[[], tuple[float, ...]],
        steps: int,
    ) -> None:
        self._pipeline = pipeline
        self._simulate = simulate
        self._get_time = time_distribution
        self._get_params = params_distribution
        self._steps = steps

        self._times = np.empty((0, self._steps + 1))
        self._params = np.empty((0, len(params_distribution())))

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            t = 0
            params = self._get_params()

            self._times = np.vstack((self._times, np.zeros((1, self._steps + 1))))
            self._params = np.vstack((self._params, np.array([params])))

            yield ImageData(image=item.image, labels=(*item.labels, str(0)))

            for i in range(1, self._steps + 1):
                t += self._get_time()
                self._times[-1, i] = t
                image = self._simulate(item.image, t, params)

                yield ImageData(image=image, labels=(*item.labels, str(i)))

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def params(self) -> np.ndarray:
        return self._params
