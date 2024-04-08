import re

from datetime import datetime
from typing import Sequence


class BrainTumorProgressionLabels:
    _subject = re.compile(r"PGBM-\d{3}")
    _date = re.compile(r"(\d{2}-\d{2}-\d{4})")
    _study = re.compile(r"\d+\.\d+-(\w+)-\d+")

    _patterns = [_subject, _date, _study]

    def __init__(self, study_labels: Sequence[str] | None = None) -> None:
        self._is_target_study = lambda s: (
            True if study_labels is None else s in study_labels
        )

    def get_label(self, dirname: str) -> str:
        for pattern in BrainTumorProgressionLabels._patterns:
            match = pattern.search(dirname)

            if match is None:
                continue

            result = match.group(pattern.groups)

            if pattern == BrainTumorProgressionLabels._date:
                date = datetime.strptime(result, "%m-%d-%Y")
                result = date.strftime("%Y-%m-%d")

            elif (
                pattern == BrainTumorProgressionLabels._study
                and not self._is_target_study(result)
            ):
                continue

            return result

        return str()
