import numpy as np

from phynn.pde.base import PDE


class FisherKolmogorovPDE(PDE):
    def _diff(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        D, p = params[:, 0], params[:, 1]

        diffusion_term = D * sum(
            [
                np.gradient(np.gradient(x, axis=dim), axis=dim)
                for dim in range(2, len(x.shape))
            ]
        )

        reaction_term = p * x * (1 - x)

        diff = diffusion_term + reaction_term

        return diff
