from abc import ABC, abstractmethod
from typing import Iterator

from imageprep import ImageData


class Pipeline(ABC):
    @abstractmethod
    def run(self) -> Iterator[ImageData]:
        raise NotImplementedError()
