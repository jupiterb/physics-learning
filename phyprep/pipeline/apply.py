import numpy as np

from abc import ABC, abstractmethod
from scipy import ndimage
from typing import Callable, Iterator

from phyprep import ImageData
from phyprep.pipeline.base import Pipeline


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


class NormalizeImages(ApplyOnImages):
    def __init__(self, pipeline: Pipeline, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(pipeline, self._normalize)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        min_value = np.min(image)
        max_value = np.max(image)
        image = (image - min_value) / (max_value - min_value) * self._scale
        return image


class Rescale(ApplyOnImages):
    def __init__(self, pipeline: Pipeline, dim: int, target_size: int) -> None:
        self._dim = dim
        self._target_size = target_size
        super().__init__(pipeline, self._rescale)

    def _rescale(self, image: np.ndarray) -> np.ndarray:
        current_size = image.shape[self._dim]

        if current_size == self._target_size:
            return image

        scaling_factor = self._target_size / current_size

        scale_factors = [1] * self._dim
        scale_factors += [scaling_factor] * (len(image.shape) - self._dim)

        image = ndimage.zoom(image, scale_factors, order=1)

        return image


class _Resize(ApplyOnImages, ABC):
    def __init__(self, pipeline: Pipeline, dim: int, target_size: int) -> None:
        self._dim = dim
        self._target_size = target_size
        super().__init__(pipeline, self._resize)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        current_size = image.shape[self._dim]

        if self._target_size == current_size:
            return image
        elif self._target_size < current_size:
            return self._cut(image)
        else:
            return self._fill(image)

    def _fill(self, image: np.ndarray) -> np.ndarray:
        pad_width: list[tuple[int, int]] = [(0, 0)] * len(image.shape)
        pad_width[self._dim] = self._padding(image)
        return np.pad(image, pad_width, mode="constant")

    @abstractmethod
    def _cut(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _padding(self, image: np.ndarray) -> tuple[int, int]:
        raise NotImplementedError()


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


class CentreOfMass(ApplyOnImages):
    def __init__(self, pipeline: Pipeline, dim: int) -> None:
        self._dim = dim
        super().__init__(pipeline, self._centre_mass_split)

    def _centre_mass_split(self, image: np.ndarray) -> np.ndarray:
        index = CentreOfMass._centre(image, self._dim)
        slices = [slice(None)] * len(image.shape)
        slices[self._dim] = slice(index, index + 1)
        indices = tuple(slices)
        return np.expand_dims(image[indices].squeeze(self._dim), 0)

    @staticmethod
    def _centre(image: np.ndarray, dim: int) -> int:
        shape = image.shape
        indices = np.indices(shape)
        return int(np.average(indices[dim], weights=image, axis=None))
