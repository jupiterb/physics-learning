import numpy as np

from typing import Iterator, Sequence

from phynn.dataloaders.base import BaseDataLoader


class HDF5DataLoader(BaseDataLoader):
    def __init__(self) -> None:
        super().__init__()

    @property
    def image_shape(self) -> Sequence[int]:
        return ()

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[np.ndarray, float]:
        return super().__getitem__(index)

    def __iter__(self) -> Iterator[int]:
        return super().__iter__()
