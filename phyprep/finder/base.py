import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


class DataFinder(ABC):
    @abstractmethod
    def find(self) -> Sequence[tuple[tuple[str, ...], Path]]:
        raise NotImplementedError()


class FileSystemFinder(DataFinder, ABC):
    def __init__(self, path: os.PathLike) -> None:
        self._path = path

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

    @abstractmethod
    def _get_label(self, name: str) -> str:
        raise NotImplementedError()
