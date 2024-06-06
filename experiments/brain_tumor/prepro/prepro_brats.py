import nibabel as nib
import numpy as np
import os
import re

import torch as th
import torchvision.transforms as T

from pathlib import Path
from typing import Sequence

from phynn.data.img import ImagesDataInterface, ImagesDataInterfaceWrapper, save


class BRATSDataInterface(ImagesDataInterface):
    def __init__(self, root: os.PathLike, labels: Sequence[str]) -> None:
        self._paths: list[list[os.PathLike]] = []

        pattern = re.compile(r"BraTS20_Training_(\d{3})_([a-zA-Z0-9]+)\.nii")

        for path, _, files in os.walk(root):
            filepaths = []

            for file in files:
                match = pattern.search(file)

                if match is None:
                    break

                label = match.group(pattern.groups)

                if label in labels:
                    filepath = os.path.join(path, file)
                    filepaths.append(filepath)

                filepaths = sorted(filepaths)

            if len(filepaths) == len(labels):
                self._paths.append(filepaths)

    @property
    def size(self) -> int:
        return len(self._paths)

    @property
    def image_shape(self) -> Sequence[int]:
        return self.get_batch([0]).shape[1:]

    def get_batch(self, ixs: Sequence[int] | th.Tensor) -> th.Tensor:
        batch = []

        for i in ixs:
            images = [
                th.from_numpy(np.asarray(nib.load(filepath).dataobj, dtype=np.float32))  # type: ignore
                for filepath in self._paths[i]
            ]

            subject = th.stack(images)
            batch.append(subject)

        return th.stack(batch)


def _max_tissue_concentration_split(batch: th.Tensor) -> th.Tensor:
    new_batch = []

    for subject in batch:
        concentration = subject[1].numpy()
        indices = np.indices(concentration.shape)
        index = int(np.average(indices[-1], weights=concentration, axis=None))
        max_concentration = subject[:, :, :, index]
        new_batch.append(max_concentration)

    return th.stack(new_batch)


def _min_max_normalize(batch: th.Tensor) -> th.Tensor:
    batch_size = batch.shape[0]
    num_channels = batch.shape[1]

    min_vals = batch.view(batch_size, num_channels, -1).min(dim=2, keepdim=True)[0]
    min_vals = min_vals.view(batch_size, num_channels, 1, 1)

    max_vals = batch.view(batch_size, num_channels, -1).max(dim=2, keepdim=True)[0]
    max_vals = max_vals.view(batch_size, num_channels, 1, 1)

    range_vals = max_vals - min_vals

    return (batch - min_vals) / range_vals


class _WeightedMean:
    def __init__(self, weight: Sequence[float]) -> None:
        self._weights = th.Tensor(weight)

    def __call__(self, batch: th.Tensor) -> th.Tensor:
        num_channels = batch.shape[1]
        return (batch * self._weights.view(1, num_channels, 1, 1)).sum(
            dim=1, keepdim=True
        )


def preprocess(
    brats_data: ImagesDataInterface, image_weights: Sequence[float]
) -> ImagesDataInterface:
    x, y = brats_data.image_shape[1:-1]

    data = ImagesDataInterfaceWrapper(brats_data, _max_tissue_concentration_split)
    data = ImagesDataInterfaceWrapper(data, _min_max_normalize)
    data = ImagesDataInterfaceWrapper(data, _WeightedMean(image_weights))
    data = ImagesDataInterfaceWrapper(data, T.Resize((x // 2, y // 2)))
    data = ImagesDataInterfaceWrapper(data, T.GaussianBlur((3, 3)))

    return data


def main() -> None:
    image_weights = {"flair": 0.0, "seg": 0.8, "t1": 0.2, "t1ce": 0.0, "t2": 0.0}
    source_path = Path(
        "./../data/raw/BRATS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    )

    brats_data = BRATSDataInterface(source_path, list(image_weights.keys()))
    preprocessed = preprocess(brats_data, list(image_weights.values()))

    result_path = Path("./data/processed/BRATS2020/result.h5")

    save(preprocessed, result_path, 64)


if __name__ == "__main__":
    main()
