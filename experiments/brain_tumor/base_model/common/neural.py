import torch.nn as nn

from phynn.nn import ResNet, ResBlockParams, Conv, ConvInitParams, ConvBlockParams
from phynn.train import training_device


def create_resnet() -> nn.Module:
    down_sampling_resnet = (
        ResNet(Conv())
        .init((ConvInitParams(1)))
        .append(ConvBlockParams(64, 3, rescale=3))
        .append(ResBlockParams(ConvBlockParams(64, 3), 3))
        .append(ResBlockParams(ConvBlockParams(64, 3), 3))
        .build()
    )

    up_sampling = (
        Conv(transpose=True)
        .init(ConvInitParams(1))
        .prepend(ConvBlockParams(64, 3, rescale=3, activation=nn.Hardtanh))
        .build()
    )

    return nn.Sequential(down_sampling_resnet, up_sampling).to(training_device)
