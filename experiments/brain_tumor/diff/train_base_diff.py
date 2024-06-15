import argparse
from dataclasses import dataclass

from common.dataset import get_data, create_dataset
from common.neural import create_u_net
from common.training import run_training


@dataclass
class Arguments:
    epochs: int
    batch_size: int
    unet_num_levels: int
    unet_initial_channels: int


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--unet_num_levels", type=int, default=3, help="Number of levels in the U-Net"
    )
    parser.add_argument(
        "--unet_initial_channels",
        type=int,
        default=64,
        help="Initial number of channels in the U-Net",
    )

    args = parser.parse_args()

    return Arguments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        unet_num_levels=args.unet_num_levels,
        unet_initial_channels=args.unet_initial_channels,
    )


def main() -> None:
    args = parse_arguments()

    train_ics, test_ics = get_data()
    train_ds = create_dataset(train_ics)
    test_ds = create_dataset(test_ics)

    net = create_u_net(args.unet_num_levels, args.unet_initial_channels)

    run_name = f"Base Diff Equation - BatchSize({args.batch_size}) - UNet({args.unet_num_levels} levels, {args.unet_initial_channels} base channels)"

    run_training(
        diff_eq_components_net=net,
        train_ds=train_ds,
        test_ds=test_ds,
        run_name=run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
