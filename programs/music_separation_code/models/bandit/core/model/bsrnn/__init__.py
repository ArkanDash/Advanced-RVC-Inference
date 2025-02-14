from abc import ABC
from typing import Iterable, Mapping, Union

from torch import nn

from models.bandit.core.model.bsrnn.bandsplit import BandSplitModule
from models.bandit.core.model.bsrnn.tfmodel import (
    SeqBandModellingModule,
    TransformerTimeFreqModule,
)


class BandsplitCoreBase(nn.Module, ABC):
    band_split: nn.Module
    tf_model: nn.Module
    mask_estim: Union[nn.Module, Mapping[str, nn.Module], Iterable[nn.Module]]

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def mask(x, m):
        return x * m
