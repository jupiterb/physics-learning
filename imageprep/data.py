from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ImageData:
    image: np.ndarray
    labels: tuple[str, ...]

    def copy(self) -> ImageData:
        return ImageData(image=self.image, labels=self.labels)

    def set_image(self, image: np.ndarray) -> ImageData:
        self.image = image
        return self
