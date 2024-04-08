import h5py
import os

from imageprep.pipeline import BasePipeline


class HDF5Exporter:
    def export(self, path: os.PathLike, pipeline: BasePipeline) -> None:
        data = pipeline.run()
        item = next(data, None)

        if item is None:
            raise ValueError("The processes did not return any item.")

        if next(data, None) is not None:
            raise ValueError(
                "Only the result of processes that return one item can be exported."
            )

        dir_path = os.path.dirname(path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=item.image)
            file.create_dataset("times", data=item.time)

            for dim, names in item.dim_names.items():
                group = file.create_group(str(dim))
                utf8_names = [name.encode("utf-8") for name in names]
                group.create_dataset(name="names", data=utf8_names)
