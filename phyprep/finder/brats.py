import os
import re

from typing import Sequence

from phyprep.finder.base import FileSystemFinder


class BRATS2020DataFinder(FileSystemFinder):
    _subject = re.compile(r"BraTS20_Training_(\d{3})\b")
    _image = re.compile(r"BraTS20_Training_(\d{3})_([a-zA-Z0-9]+)\.nii")

    _patterns = [_subject, _image]

    def __init__(
        self, path: os.PathLike, study_labels: Sequence[str] | None = None
    ) -> None:
        super().__init__(path)
        self._is_target_study = lambda s: (
            True if study_labels is None else s in study_labels
        )

    def _get_label(self, name: str) -> str:
        for pattern in BRATS2020DataFinder._patterns:
            match = pattern.search(name)

            if match is None:
                continue

            result = match.group(pattern.groups)

            if pattern == BRATS2020DataFinder._image and not self._is_target_study(
                result
            ):
                continue

            return result

        return str()
