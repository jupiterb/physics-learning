import numpy as np

from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import BasePipeline


class NormalizeImages(BasePipeline):
    def __init__(self, pipeline: BasePipeline, scale: float = 1.0) -> None:
        self._pipeline = pipeline
        self._scale = scale

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            image = item.image

            min_value = np.min(image)
            max_value = np.max(image)

            image = (image - min_value) / (max_value - min_value) * self._scale

            yield item.copy().set_image(image)
