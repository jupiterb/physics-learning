from common.dataset import get_data, create_dataset
from common.neural import create_resnet
from common.training import run_training


def main() -> None:
    train_ics, test_ics = get_data()
    train_ds = create_dataset(train_ics)
    test_ds = create_dataset(test_ics)

    diffusion_net = create_resnet()
    proliferation_net = create_resnet()

    run_training(
        neural_nets=[diffusion_net, proliferation_net],
        train_ds=train_ds,
        test_ds=test_ds,
        run_name="equation_diffusion_proliferation_together",
        epochs=50,
    )


if __name__ == "__main__":
    main()
