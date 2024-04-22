import numpy as np

from abc import ABC, abstractmethod
from typing import Iterator, Mapping

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class _Aggregate(Pipeline, ABC):
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def run(self) -> Iterator[ImageData]:
        to_stack: list[ImageData] = []
        group_labels: tuple[str, ...] = ()

        for item in self._pipeline.run():
            if len(to_stack) == 0 or group_labels == item.labels[:-1]:
                to_stack.append(item)
            else:
                yield self._aggregate(to_stack)
                to_stack = [item]

            group_labels = item.labels[:-1]

        yield self._aggregate(to_stack)

    @abstractmethod
    def _aggregate(self, data: list[ImageData]) -> ImageData:
        raise NotADirectoryError()


class StackImages(_Aggregate):
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__(pipeline)

    def _aggregate(self, data: list[ImageData]) -> ImageData:
        images = [item.image for item in data]
        image = np.stack(images)

        labels = data[0].labels[:-1]

        return ImageData(
            image=image,
            labels=labels,
        )


class MeanOfImages(StackImages):
    def __init__(self, pipeline: Pipeline, weights: Mapping[str, float]) -> None:
        super().__init__(pipeline)
        self._weights = weights

    def _aggregate(self, data: list[ImageData]) -> ImageData:
        stacked = super()._aggregate(data)
        image = stacked.image

        dim_names = (item.labels[-1] for item in data)
        weights = np.array([self._weights[name] for name in dim_names])

        weights_shape = [len(self._weights)] + [1] * (image.ndim - 1)
        weights = weights.reshape(weights_shape)

        image = image * weights
        image = np.sum(image, axis=0)

        return stacked.set_image(image)
