import argparse
import os

from phyprep.export import HDF5Exporter
from phyprep.finder import BrainTumorProgressionDataFinder
from phyprep.pipeline import (
    DICOMImagesLoader,
    MeanOfImages,
    NormalizeImages,
    Rescale,
    StackImages,
    TakeCentre,
    TakeFromBegin,
)
from phyprep.utils import get_min_shape, get_time_data


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess brain tumor images.")

    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("target_path", type=str, help="Path to the target directory")

    parser.add_argument(
        "--target_length",
        type=int,
        default=128,
        required=False,
        help="Target length of images after preprocessing",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=128,
        required=False,
        help="Target width of images after preprocessing",
    )

    return parser.parse_args()


def preprocess(
    dataset_path: os.PathLike,
    target_path: os.PathLike,
    target_length: int,
    target_width: int,
) -> None:
    time_format = "%Y-%m-%d"
    image_type_weights = {
        "T1post": 0.06,
        "T2reg": 0.06,
        "MaskTumor": 0.7,
        "FLAIRreg": 0.06,
        "ADCreg": 0.06,
        "T1prereg": 0.06,
    }

    finder = BrainTumorProgressionDataFinder(
        dataset_path, list(image_type_weights.keys())
    )

    pipeline = DICOMImagesLoader(finder)
    pipeline = StackImages(pipeline)

    target_height, _, _ = get_min_shape(pipeline.run())

    pipeline = TakeFromBegin(pipeline, 0, target_height)
    pipeline = Rescale(pipeline, 1, target_length)
    pipeline = TakeCentre(pipeline, 2, target_width)
    pipeline = NormalizeImages(pipeline)
    pipeline = MeanOfImages(pipeline, weights=image_type_weights)
    pipeline = StackImages(pipeline)
    pipeline = StackImages(pipeline)

    image = next(pipeline.run()).image
    time = get_time_data(finder, 1, time_format)

    exporter = HDF5Exporter()
    exporter.export(target_path, image, time)


def main():
    args = get_args()
    preprocess(
        args.dataset_path, args.target_path, args.target_length, args.target_width
    )


if __name__ == "__main__":
    main()
