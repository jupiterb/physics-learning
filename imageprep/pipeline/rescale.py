from scipy import ndimage
from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class Rescale(Pipeline):

    def __init__(self, pipeline: Pipeline, dim: int, target_size: int) -> None:
        self._pipeline = pipeline
        self._dim = dim
        self._target_size = target_size

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            current_size = item.image.shape[self._dim]

            if current_size == self._target_size:
                yield item
                continue

            scaling_factor = self._target_size / current_size

            scale_factors = [1] * self._dim
            scale_factors += [scaling_factor] * (len(item.image.shape) - self._dim)

            image = ndimage.zoom(item.image, scale_factors, order=1)

            yield item.copy().set_image(image)
