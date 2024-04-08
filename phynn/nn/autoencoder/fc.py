from typing import Sequence

from phynn.nn.autoencoder.base import AutoEncoder
from phynn.nn.elementary.activation import ActivationFunction
from phynn.nn.elementary.fc import FC


class FCAutoEncoder(AutoEncoder):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_functions: Sequence[ActivationFunction] | None = None,
        inner: AutoEncoder | None = None,
    ) -> None:
        encoder = FC(layer_sizes, activation_functions)
        decoder = FC(
            list(reversed(layer_sizes)),
            (
                None
                if activation_functions is None
                else list(reversed(activation_functions))
            ),
        )
        super().__init__(encoder, decoder, inner)
