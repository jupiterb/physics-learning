import numpy as np

from phynn.pde.reaction_diffusion import ReactionDiffusionPDE


class ReactionDiffusionAdvectionPDE(ReactionDiffusionPDE):
    def _diff(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        diff = super()._diff(x, params)

        v = params[:, 2].reshape(x.shape)

        advection_term = v * sum(
            [np.gradient(x, axis=dim) for dim in range(2, len(x.shape))]
        )

        diff -= advection_term

        return diff
