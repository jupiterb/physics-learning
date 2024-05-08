import numpy as np

from typing import Callable, Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class GenerateImages(Pipeline):
    def __init__(
        self,
        pipeline: Pipeline,
        generator: Callable[[np.ndarray], Iterator[np.ndarray]],
        replace_label: bool,
    ) -> None:
        self._pipeline = pipeline
        self._generator = generator
        self._replace_label = replace_label

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            for i, image in enumerate(self._generator(item.image)):
                labels = (
                    (*item.labels[:-1], f"{item.labels[-1]}-{i}")
                    if self._replace_label
                    else (*item.labels, str(i))
                )
                yield ImageData(image=image, labels=labels)


class Simulate(GenerateImages):
    def __init__(
        self,
        pipeline: Pipeline,
        simulate: Callable[[np.ndarray, int, tuple[float, ...]], np.ndarray],
        time_distribution: Callable[[], int],
        params_distribution: Callable[[], tuple[float, ...]],
        steps: int,
    ) -> None:
        self._simulate = simulate
        self._get_time = time_distribution
        self._get_params = params_distribution
        self._steps = steps

        self._times = np.empty((0, self._steps + 1))
        self._params = np.empty((0, len(params_distribution())))

        super().__init__(pipeline, self._simulation, False)

    def _simulation(self, image: np.ndarray) -> Iterator[np.ndarray]:
        yield image

        t = 0
        params = self._get_params()

        self._times = np.vstack((self._times, np.zeros((1, self._steps + 1))))
        self._params = np.vstack((self._params, np.array([params])))

        for i in range(1, self._steps + 1):
            t_diff = self._get_time()
            image = self._simulate(image, t_diff, params)
            t += t_diff
            self._times[-1, i] = t
            yield image

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def params(self) -> np.ndarray:
        return self._params


class Repeat(GenerateImages):
    def __init__(self, pipeline: Pipeline, repeats: int) -> None:
        self._repeats = repeats
        super().__init__(pipeline, self._repeat, True)

    def _repeat(self, image: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(self._repeats):
            yield image
