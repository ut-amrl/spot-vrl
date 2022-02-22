from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from spot_vrl.data import ImuData


class SingleTerrainDataset(Dataset[torch.Tensor]):
    """IMU dataset from a single terrain type.

    Datum are organized as fixed-sized matrices where rows encode individual
    data types over time and columns encode observations of all data types.
    """

    def __init__(
        self, path: Union[str, Path], start: float = 0, end: float = np.inf
    ) -> None:
        imu = ImuData(path)

        window_size = 40  # approx 3 seconds at 13.6 Hz
        self.windows: List[torch.Tensor] = []

        ts, data = imu.query_time_range(imu.all_sensor_data, start, end)
        # Overlap windows for more data points
        for i in range(0, data.shape[-1] - window_size + 1, window_size // 5):
            window = data[:, i : i + window_size]

            # Add statistical features as additional column vectors
            mean = window.mean(axis=1)[:, np.newaxis]
            std = window.std(axis=1)[:, np.newaxis]
            med = np.median(window, axis=1)[:, np.newaxis]
            # TODO(eyang): Joint data is periodic, so these features may not be
            # very useful. May want to use quantiles, IQR, min, max, etc.

            window = np.hstack((window, mean, std, med))
            self.windows.append(torch.from_numpy(window))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


Triplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ManualTripletDataset(Dataset[Triplet]):
    """Hardcoded triplet dataset using fixed log files and time ranges."""

    def __init__(self) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Triplet:
        ...
