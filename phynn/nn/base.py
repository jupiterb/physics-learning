from __future__ import annotations

import torch.nn as nn

from abc import ABC, abstractmethod
from typing import TypeVar, Generic


NNInitParams = TypeVar("NNInitParams")
NNBlockParams = TypeVar("NNBlockParams")


class NNBuilder(Generic[NNInitParams, NNBlockParams], ABC):
    @abstractmethod
    def init(self, params: NNInitParams) -> NNBuilder[NNInitParams, NNBlockParams]:
        raise NotImplementedError()

    @abstractmethod
    def prepend(self, params: NNBlockParams) -> NNBuilder[NNInitParams, NNBlockParams]:
        raise NotImplementedError()

    @abstractmethod
    def append(self, params: NNBlockParams) -> NNBuilder[NNInitParams, NNBlockParams]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, keep_end: bool) -> NNBuilder[NNInitParams, NNBlockParams]:
        raise NotImplementedError()

    @abstractmethod
    def build(self) -> nn.Sequential:
        raise NotImplementedError()
