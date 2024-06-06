from phynn.diff.terms import DiffusionTerm, ProliferationTerm

from common.dataset import get_data, create_dataset
from common.neural import create_u_net
from common.training import run_training


def main() -> None:
    # set up datasets

    train_ics, test_ics = get_data()
    train_ds = create_dataset(train_ics)
    test_ds = create_dataset(test_ics)

    # train diffusion only

    diffusion_net = create_u_net()

    run_training(
        neural_nets=[diffusion_net, ProliferationTerm()],
        train_ds=train_ds,
        test_ds=test_ds,
        run_name="equation_diffusion_only",
        epochs=40,
    )

    # train proliferation only

    proliferation_net = create_u_net()

    run_training(
        neural_nets=[DiffusionTerm(), proliferation_net],
        train_ds=train_ds,
        test_ds=test_ds,
        run_name="equation_proliferation_only",
        epochs=40,
    )

    # fine tune together

    run_training(
        neural_nets=[diffusion_net, proliferation_net],
        train_ds=train_ds,
        test_ds=test_ds,
        run_name="equation_diffusion_proliferation_fine_tune",
        epochs=10,
        lr=0.00005,
    )


if __name__ == "__main__":
    main()
