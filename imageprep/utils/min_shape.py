from imageprep import ImageData

from typing import Iterator, Sequence


def get_min_shape(data: Iterator[ImageData]) -> Sequence[int]:
    shapes = [item.image.shape for item in data]
    min_sizes = [min(sizes) for sizes in zip(*shapes)]
    return min_sizes
