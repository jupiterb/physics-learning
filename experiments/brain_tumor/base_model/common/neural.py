import torch.nn as nn

from phynn.nn import ResBlock, Conv, ConvBlockParams
from phynn.train import training_device


def create_u_net() -> nn.Module:
    inner = (
        Conv(48)
        .append(ConvBlockParams(64, 3, rescale=2))
        .transpose(True)  # type: ignore
        .append(ConvBlockParams(48, 3, rescale=2))
    )
    inner_res = ResBlock(inner.unload())

    down_sampling = Conv(32).append(ConvBlockParams(48, 3, rescale=2, dropout=0.1))
    up_sampling = Conv(32, transpose=True).prepend(ConvBlockParams(48, 3, rescale=2))

    middle = ResBlock(
        nn.Sequential(down_sampling.unload(), inner_res, up_sampling.unload())
    )

    down_sampling.prepend(ConvBlockParams(1, 3, rescale=2, dropout=0.3))
    up_sampling.append(ConvBlockParams(1, 3, nn.Hardtanh, rescale=2))

    return ResBlock(
        nn.Sequential(down_sampling.unload(), middle, up_sampling.unload())
    ).to(training_device)
