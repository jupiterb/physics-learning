import numpy as np

from dataclasses import dataclass


@dataclass
class ImageData:
    image: np.ndarray
    labels: tuple[str, ...]
    dim_names: dict[int, list[str]]
    time: np.ndarray | None = None
