import h5py
import numpy as np
import os


class HDF5Exporter:
    def export(self, path: os.PathLike, image: np.ndarray, time: np.ndarray) -> None:
        dir_path = os.path.dirname(path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=image)
            file.create_dataset("times", data=time)
