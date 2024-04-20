import os
import re

from datetime import datetime
from typing import Sequence

from imageprep.finder.base import FileSystemFinder


class BrainTumorProgressionDataFinder(FileSystemFinder):
    _subject = re.compile(r"PGBM-\d{3}")
    _date = re.compile(r"(\d{2}-\d{2}-\d{4})")
    _study = re.compile(r"\d+\.\d+-(\w+)-\d+")
    _image = re.compile(r"1-(\d{2})\.dcm")

    _patterns = [_subject, _date, _study, _image]

    def __init__(
        self, path: os.PathLike, study_labels: Sequence[str] | None = None
    ) -> None:
        super().__init__(path)
        self._is_target_study = lambda s: (
            True if study_labels is None else s in study_labels
        )

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
