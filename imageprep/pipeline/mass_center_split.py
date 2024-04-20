import numpy as np

from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class CenterOfMassSplit(Pipeline):
    def __init__(self, pipeline: Pipeline, dim: int) -> None:
        self._pipeline = pipeline
        self._dim = dim

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            index = CenterOfMassSplit._centre(item.image, self._dim)

            slices = [slice(None)] * len(item.image.shape)
            slices[self._dim] = slice(index, index + 1)
            indices = tuple(slices)

            image = np.expand_dims(item.image[indices].squeeze(self._dim), 0)

            yield item.copy().set_image(image)

    @staticmethod
    def _centre(image: np.ndarray, dim: int) -> int:
        shape = image.shape
        indices = np.indices(shape)
        return int(np.average(indices[dim], weights=image, axis=None))
