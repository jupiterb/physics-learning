import nibabel as nib
import numpy as np
import pydicom

from typing import Iterator

from phyprep import ImageData
from phyprep.finder import DataFinder
from phyprep.pipeline.base import Pipeline


class DICOMImagesLoader(Pipeline):
    def __init__(self, finder: DataFinder) -> None:
        self._finder = finder

    def run(self) -> Iterator[ImageData]:
        for labels, filepath in self._finder.find():
            image = pydicom.dcmread(filepath).pixel_array

            if image is not None:
                yield ImageData(image=image, labels=labels)


class NIfTIImagesLoader(Pipeline):
    def __init__(self, finder: DataFinder) -> None:
        self._finder = finder

    def run(self) -> Iterator[ImageData]:
        for labels, filepath in self._finder.find():
            image = np.asarray(nib.load(filepath).dataobj)  # type: ignore

            if image is not None:
                yield ImageData(image=image, labels=labels)
