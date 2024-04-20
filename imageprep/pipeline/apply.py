import numpy as np

from typing import Callable, Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class ApplyOnImages(Pipeline):
    def __init__(
        self, pipeline: Pipeline, operation: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self._pipeline = pipeline
        self._operation = operation

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            image = self._operation(item.image)
            yield item.copy().set_image(image)
