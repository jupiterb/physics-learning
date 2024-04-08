import numpy as np

from datetime import datetime
from typing import Iterator

from imageprep import ImageData
from imageprep.pipeline.base import BasePipeline


class StackImages(BasePipeline):
    def __init__(self, pipeline: BasePipeline) -> None:
        self._pipeline = pipeline

    def run(self) -> Iterator[ImageData]:
        to_stack: list[ImageData] = []
        group_labels: tuple[str, ...] = ()

        for item in self._pipeline.run():
            if len(to_stack) == 0 or group_labels == item.labels[:-1]:
                to_stack.append(item)
            else:
                yield self._stack(to_stack)
                to_stack = [item]

            group_labels = item.labels[:-1]

        yield self._stack(to_stack)

    def _stack(self, data: list[ImageData]) -> ImageData:
        images = [item.image for item in data]
        image = np.stack(images)

        time = self._get_time(data)

        labels = data[0].labels[:-1]

        dim_names = data[0].dim_names.copy()
        dim_names[len(labels)] = [item.labels[-1] for item in data]

        return ImageData(
            image=image,
            labels=labels,
            dim_names=dim_names,
            time=time,
        )

    def _get_time(self, data: list[ImageData]) -> np.ndarray | None:
        times = [item.time for item in data if item.time is not None]
        return np.stack(times) if len(times) > 0 else None


class StackImagesAsTimeInterval(StackImages):
    def __init__(self, source: BasePipeline, time_format: str) -> None:
        super().__init__(source)
        self._time_format = time_format

    def _stack(self, data: list[ImageData]) -> ImageData:
        stacked = super()._stack(data)
        stacked.dim_names[len(stacked.labels)] = [
            str(i) for i in range(len(stacked.image))
        ]
        return stacked

    def _get_time(self, data: list[ImageData]) -> np.ndarray | None:
        dates = [datetime.strptime(item.labels[-1], self._time_format) for item in data]
        days = [(date - dates[0]).days for date in dates]
        return np.array(days)
