import os
import re

from datetime import datetime
from pathlib import Path
from typing import Sequence

from imageprep.finder.base import BaseDataFinder


class BrainTumorProgressionDataFinder(BaseDataFinder):
    _subject = re.compile(r"PGBM-\d{3}")
    _date = re.compile(r"(\d{2}-\d{2}-\d{4})")
    _study = re.compile(r"\d+\.\d+-(\w+)-\d+")
    _image = re.compile(r"1-(\d{2})\.dcm")

    _patterns = [_subject, _date, _study, _image]

    def __init__(
        self, path: os.PathLike, study_labels: Sequence[str] | None = None
    ) -> None:
        self._path = path
        self._is_target_study = lambda s: (
            True if study_labels is None else s in study_labels
        )

    def find(self) -> Sequence[tuple[tuple[str, ...], Path]]:
        labeled_paths: list[tuple[tuple[str, ...], Path]] = []

        for path, _, files in os.walk(self._path):
            filepaths = (os.path.join(path, file) for file in files)

            for filepath in filepaths:
                separated = str(Path(filepath).relative_to(self._path)).split(os.sep)
                labels = tuple(map(self._get_label, separated))

                if all(labels):
                    labeled_paths.append((labels, filepath))

        return sorted(labeled_paths, key=lambda pair: pair[0])

    def _get_label(self, name: str) -> str:
        for pattern in BrainTumorProgressionDataFinder._patterns:
            match = pattern.search(name)

            if match is None:
                continue

            result = match.group(pattern.groups)

            if pattern == BrainTumorProgressionDataFinder._date:
                date = datetime.strptime(result, "%m-%d-%Y")
                result = date.strftime("%Y-%m-%d")

            elif (
                pattern == BrainTumorProgressionDataFinder._study
                and not self._is_target_study(result)
            ):
                continue

            return result

        return str()
