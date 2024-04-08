import argparse
import os

from typing import Sequence

from imageprep.labels import BrainTumorProgressionLabels
from imageprep.pipeline import (
    DicomImagesLoader,
    FromBegin,
    Centre,
    Rescale,
    StackImages,
    StackImagesAsTimeInterval,
)
from imageprep.export import HDF5Exporter
from imageprep import utils


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess brain tumor images.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("target_path", type=str, help="Path to the target directory")
    parser.add_argument(
        "--image_types",
        nargs="+",
        default=["T1post", "T2reg", "MaskTumor", "FLAIRreg", "ADCreg", "T1prereg"],
        help="List of image types to include in preprocessing",
    )
    return parser.parse_args()


def preprocess(
    dataset_path: os.PathLike, target_path: os.PathLike, image_types: Sequence[str]
) -> None:
    data_labels = BrainTumorProgressionLabels(image_types)

    loader = DicomImagesLoader(dataset_path, data_labels.get_label)

    long, height, width = utils.get_target_shape(loader.run())
    time_format = "%Y-%m-%d"

    pipeline = FromBegin(loader, 0, long)
    pipeline = Rescale(pipeline, 1, height)
    pipeline = Centre(pipeline, 2, width)
    pipeline = StackImages(pipeline)
    pipeline = StackImagesAsTimeInterval(pipeline, time_format)
    pipeline = StackImages(pipeline)

    exporter = HDF5Exporter()
    exporter.export(target_path, pipeline)


def main():
    args = get_args()
    preprocess(args.dataset_path, args.target_path, args.image_types)


if __name__ == "__main__":
    main()
