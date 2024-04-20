from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import Pipeline


class Reply(Pipeline):
    def __init__(self, pipeline: Pipeline, factor: int, replace_label: bool) -> None:
        self._pipeline = pipeline
        self._factor = factor
        self._replace_label = replace_label

    def run(self) -> Iterator[ImageData]:
        for i, item in enumerate(self._pipeline.run()):
            labels = item.labels

            for j in range(self._factor):
                labels = (
                    (*labels[:-1], str(j)) if self._replace_label else (*labels, str(j))
                )
                yield item.copy().set_labels(labels)
