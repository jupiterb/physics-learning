import torch.optim as optim

from pathlib import Path

from phynn.data.img import (
    ImagesDataInterface,
    HDF5ImagesDataInterface,
    ImagesDataset,
    train_test_split,
)
from phynn.models import VAEModel, OptimizerParams
from phynn.nn import (
    VariationalAutoEncoder,
    AutoEncoderBuilder,
    Conv,
    ConvInitParams,
    ConvBlockParams,
    FC,
    FCInitParams,
    FCBlockParams,
)
from phynn.train import train, training_device


def get_data() -> tuple[ImagesDataInterface, ImagesDataInterface]:
    path = Path("./../data/processed/BRATS2020/result.h5")
    all = HDF5ImagesDataInterface(path, training_device)
    return train_test_split(all, 0.8)


def create_dataset(data_interface: ImagesDataInterface) -> ImagesDataset:
    return ImagesDataset(data_interface)


def create_vae() -> VariationalAutoEncoder:
    in_shape = (1, 120, 120)
    latent_size = 64

    conv_encoder_builder = Conv()
    conv_decoder_builder = Conv(transpose=True, upsample=True, rescale_on_begin=True)

    conv_ae = (
        AutoEncoderBuilder(conv_encoder_builder, conv_decoder_builder)
        .init(in_shape, ConvInitParams(1))
        .add_block(ConvBlockParams(32, 3, rescale=3))
        .add_block(ConvBlockParams(32, 3))
        .add_block(ConvBlockParams(64, 3, rescale=2))
        .add_block(ConvBlockParams(64, 3))
        .add_block(ConvBlockParams(96, 3, rescale=2))
        .build()
    )

    fc_input_size = 96 * 10 * 10

    fc_ae = (
        AutoEncoderBuilder(FC(), FC())
        .init((fc_input_size,), FCInitParams(fc_input_size))
        .add_block(FCBlockParams(1024))
        .add_block(FCBlockParams(256))
        .build()
    )

    ae = conv_ae.flatten().add_inner(fc_ae)

    return VariationalAutoEncoder(in_shape, ae.encoder, ae.decoder, latent_size).to(
        training_device
    )


def run_training(
    vae: VariationalAutoEncoder,
    train_ds: ImagesDataset,
    test_ds: ImagesDataset,
    run_name: str,
    epochs: int,
    lr: float = 0.00003,
) -> None:
    vae_model = VAEModel(vae, optimizer_params=OptimizerParams(optim.AdamW, lr))

    train(
        vae_model,
        run_name=run_name,
        train_dataset=train_ds,
        val_dataset=test_ds,
        batch_size=64,
        epochs=epochs,
    )


def main() -> None:
    train_ics, test_ics = get_data()
    train_ds = create_dataset(train_ics)
    test_ds = create_dataset(test_ics)

    vae = create_vae()

    run_training(
        vae=vae,
        train_ds=train_ds,
        test_ds=test_ds,
        run_name="vae",
        epochs=300,
    )


if __name__ == "__main__":
    main()
