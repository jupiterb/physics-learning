import numpy as np

from abc import ABC, abstractmethod
from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class _Resize(Pipeline, ABC):
    def __init__(self, pipeline: Pipeline, dim: int, target_size: int) -> None:
        self._pipeline = pipeline
        self._dim = dim
        self._target_size = target_size

    def run(self) -> Iterator[ImageData]:
        for item in self._pipeline.run():
            current_size = item.image.shape[self._dim]

            if self._target_size == current_size:
                yield item
                continue

            if self._target_size < current_size:
                image = self._cut(item.image)
            else:
                padding = self._padding(item.image)
                image = self._fill(item.image, padding)

            yield item.copy().set_image(image)

    @abstractmethod
    def _cut(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _padding(self, image: np.ndarray) -> tuple[int, int]:
        raise NotImplementedError()

    def _fill(self, image: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
        pad_width: list[tuple[int, int]] = [(0, 0)] * len(image.shape)
        pad_width[self._dim] = padding
        return np.pad(image, pad_width, mode="constant")


class TakeFromBegin(_Resize):
    def _cut(self, image: np.ndarray) -> np.ndarray:
        slices = [slice(None)] * len(image.shape)
        slices[self._dim] = slice(0, self._target_size)
        return image[tuple(slices)]

    def _padding(self, image: np.ndarray) -> tuple[int, int]:
        return (0, self._target_size - image.shape[self._dim])


class TakeToEnd(_Resize):
    def _cut(self, image: np.ndarray) -> np.ndarray:
        slices = [slice(None)] * len(image.shape)
        slices[self._dim] = slice(-self._target_size, None)
        return image[tuple(slices)]

    def _padding(self, image: np.ndarray) -> tuple[int, int]:
        return (self._target_size - image.shape[self._dim], 0)


class TakeCentre(_Resize):
    def __init__(self, source: Pipeline, dim: int, target_size: int) -> None:
        super().__init__(source, dim, target_size)
        self._half_size = self._target_size // 2

    def _cut(self, image: np.ndarray) -> np.ndarray:
        center = image.shape[self._dim] // 2
        start = max(0, center - self._half_size)
        end = min(center + self._half_size, image.shape[self._dim])
        slice_obj = [slice(None)] * len(image.shape)
        slice_obj[self._dim] = slice(start, end)
        return image[tuple(slice_obj)]

    def _padding(self, image: np.ndarray) -> tuple[int, int]:
        padding_width = (self._target_size - image.shape[self._dim]) // 2
        return (padding_width, padding_width)
