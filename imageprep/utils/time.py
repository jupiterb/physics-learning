import numpy as np

from datetime import datetime

from imageprep.finder import DataFinder


def get_time_data(finder: DataFinder, time_dim: int, time_format: str) -> np.ndarray:
    target_labels = sorted({labels[: time_dim + 1] for labels, _ in finder.find()})

    groups: dict[tuple[str, ...], list] = {
        labels[:time_dim]: [] for labels in target_labels
    }

    for labels in target_labels:
        groups[labels[:time_dim]].append(
            datetime.strptime(labels[time_dim], time_format)
        )

    times = [[(date - dates[0]).days for date in dates] for dates in groups.values()]

    label_sets = [set() for _ in range(time_dim)]

    for labels in target_labels:
        for i, label in enumerate(labels[:-1]):
            label_sets[i].add(label)

    time_sequence_length = len(times[0])
    time_shape = [len(s) for s in label_sets] + [time_sequence_length]

    return np.array(times).reshape(time_shape)
