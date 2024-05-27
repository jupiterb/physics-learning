import torch as th
from torch.utils.data import Dataset
from typing import Sequence

from phynn.data.img.interface import ImagesDataInterface


class ImagesDataset(Dataset):
    def __init__(self, data: ImagesDataInterface) -> None:
        super().__init__()
        self._data = data

    def __len__(self) -> int:
        return self._data.size

    def __getitem__(self, index: int) -> th.Tensor:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> th.Tensor:
        return self._data.get_batch(indices)
