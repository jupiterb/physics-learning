import pydicom

from typing import Iterator

from imageprep import ImageData
from imageprep.finder import BaseDataFinder
from imageprep.pipeline.base import BasePipeline


class DicomImagesLoader(BasePipeline):
    def __init__(self, finder: BaseDataFinder) -> None:
        self._finder = finder

    def run(self) -> Iterator[ImageData]:
        for labels, filepath in self._finder.find():
            image = pydicom.dcmread(filepath).pixel_array

            if image is not None:
                yield ImageData(image=image, labels=labels)
