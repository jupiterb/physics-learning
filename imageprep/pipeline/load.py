import numpy as np
import os
import pydicom

from pathlib import Path
from typing import Callable, Iterator

from imageprep import ImageData
from imageprep.pipeline.base import BasePipeline


def _load_dicom_images_to_numpy(path: os.PathLike) -> np.ndarray | None:
    dicom_file_paths = [
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".dcm")
    ]

    dicom_files = [pydicom.dcmread(path) for path in dicom_file_paths]
    dicom_files = sorted(dicom_files, key=lambda file: file.InstanceNumber)

    dicom_images = [file.pixel_array for file in dicom_files]
    image = np.stack(dicom_images) if dicom_images else None

    return image


class DicomImagesLoader(BasePipeline):
    def __init__(self, path: os.PathLike, label_getter: Callable[[str], str]) -> None:
        self._path = path
        self._get_label = label_getter

    def run(self) -> Iterator[ImageData]:
        labeled_paths: list[tuple[tuple[str, ...], Path]] = []

        for path, _, _ in os.walk(self._path):
            dirnames = tuple(str(Path(path).relative_to(self._path)).split(os.sep))
            labels = tuple(self._get_label(dirname) for dirname in dirnames)

            if all(labels):
                labeled_paths.append((labels, path))

        labeled_paths = sorted(labeled_paths, key=lambda pair: pair[0])

        for labels, path in labeled_paths:
            image = _load_dicom_images_to_numpy(path)

            if image is not None:
                yield ImageData(image=image, labels=labels, dim_names={})
