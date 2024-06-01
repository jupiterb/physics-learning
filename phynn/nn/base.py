from __future__ import annotations

from abc import ABC, abstractmethod
from torch import nn
from typing import TypeVar, Generic


NNBlockParams = TypeVar("NNBlockParams")


class NNBuilder(Generic[NNBlockParams], ABC):
    def __init__(self) -> None:
        super(NNBuilder, self).__init__()
        self._nn = nn.Sequential()

    @property
    def nn(self) -> nn.Sequential:
        return self._nn

    def unload(self) -> nn.Sequential:
        result = self._nn
        self._nn = nn.Sequential()
        return result

    @abstractmethod
    def prepend(self, params: NNBlockParams) -> NNBuilder[NNBlockParams]:
        raise NotImplementedError()

    @abstractmethod
    def append(self, params: NNBlockParams) -> NNBuilder[NNBlockParams]:
        raise NotImplementedError()
