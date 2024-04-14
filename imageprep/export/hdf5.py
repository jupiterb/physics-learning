import h5py
import numpy as np
import os


class HDF5Exporter:
    def export(
        self,
        path: os.PathLike,
        image: np.ndarray,
        time: np.ndarray,
        time_series_params: np.ndarray | None = None,
    ) -> None:
        dir_path = os.path.dirname(path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=image)
            file.create_dataset("times", data=time)

            if time_series_params is not None:
                file.create_dataset("params", time_series_params)
