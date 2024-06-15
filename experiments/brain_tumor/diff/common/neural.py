import torch.nn as nn

from phynn.nn import UNet, Conv, ConvInitParams, ConvBlockParams
from phynn.train import training_device


def create_u_net(
    num_levels: int = 3, initial_channels: int = 64, out_channels: int = 2
) -> nn.Module:
    encoder_builder = Conv(rescale_on_begin=True)
    decoder_builder = Conv(upsample=True)

    unet_builder = UNet(encoder_builder, decoder_builder).init(
        ConvInitParams(initial_channels)
    )

    level_channels = initial_channels

    for _ in range(num_levels):
        unet_builder.split_level(
            ConvBlockParams(level_channels), ConvBlockParams(level_channels * 2)
        ).add_symmetrical_blocks(ConvBlockParams(level_channels * 2, rescale=2))

        level_channels *= 2

    unet = unet_builder.build()

    unet_in = (
        Conv().init(ConvInitParams(1)).append(ConvBlockParams(initial_channels)).build()
    )

    unet_out = (
        Conv()
        .init(ConvInitParams(initial_channels))
        .append(ConvBlockParams(initial_channels))
        .append(ConvBlockParams(out_channels, kernel_size=1, activation=nn.Hardtanh))
        .build()
    )

    return nn.Sequential(unet_in, unet, unet_out).to(training_device)
