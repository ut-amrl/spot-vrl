from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.utils.data
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset

from spot_vrl.data import ImuData


class SingleTerrainDataset(Dataset[torch.Tensor]):
    """IMU dataset from a single terrain type.

    Data are time windows as fixed-sized matrices in the layout:

          *-------> data types
          |
          |
          |
          V
        time
    """

    def __init__(
        self, path: Union[str, Path], start: float = 0, end: float = np.inf
    ) -> None:
        imu = ImuData(path)

        window_size = 40
        self.windows: List[torch.Tensor] = []

        ts, data = imu.query_time_range(imu.all_sensor_data, start, end)
        # Overlap windows for more data points
        for i in range(0, data.shape[0] - window_size + 1, window_size // 8):
            window = data[i : i + window_size]

            # compute ffts
            # ffts: List[npt.NDArray[np.float32]] = []
            # all_joint_info = window[:, 1:37]
            # for i in range(36):
            #     fft = np.fft.fft(all_joint_info[:, i])
            #     fft = np.abs(fft)
            #     fft /= np.max(fft)
            #     ffts.append(fft.astype(np.float32))

            # window = np.vstack((window, *ffts))

            # Add statistical features as additional row vectors
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            skew = stats.skew(window, axis=0)
            kurtosis = stats.kurtosis(window, axis=0)
            med = np.median(window, axis=0)
            q1 = np.quantile(window, 0.25, axis=0).astype(np.float32)
            q3 = np.quantile(window, 0.75, axis=0).astype(np.float32)
            # TODO(eyang): Joint data is periodic, so these features may not be
            # very useful. May want to use quantiles, IQR, min, max, etc.

            # window = np.vstack((window, mean, std, skew, kurtosis, med, q1, q3))
            window = np.vstack((mean, std, skew, kurtosis, med, q1, q3))
            self.windows.append(torch.from_numpy(window))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


Triplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BaseTripletDataset(Dataset[Triplet], ABC):
    """Base class for IMU Triplet Datasets

    All derived classes must overload `__len__`, optionally using a scaling
    multiplier to artificially increase the number of triplets to generate.

    Example:
    >>> class DerivedTripletDataset(BaseTripletDataset):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self._add_category("key1", (...))
    ...         self._add_category("key2", (...))
    ...         self._add_category("key3", (...))
    ...
    ...     def __len__(self) -> int:
    ...         k = len(self._categories)
    ...         k = k * (k - 1) // 2
    ...         return super().__len__() * k
    """

    def __init__(self) -> None:
        self._categories: Dict[str, ConcatDataset[torch.Tensor]] = {}
        self._cumulative_sizes: List[int] = []
        self._rng = np.random.default_rng()

    def _add_category(self, key: str, datasets: Sequence[SingleTerrainDataset]) -> None:
        self._categories[key] = ConcatDataset(datasets)
        self._cumulative_sizes = np.cumsum(
            [len(ds) for ds in self._categories.values()], dtype=np.int_
        ).tolist()

    def _get_random_datum(self, from_cats: Sequence[str] = ()) -> torch.Tensor:
        """Returns a random datum from the specified categories.

        Each category in the sequence has equal probability of being chosen.

        Args:
            from_cats (tuple[str] | list[str]): A sequence of strings containing
                the categories to sample from. If the sequence is empty, all of
                the internal categories are used instead.
        """
        if not from_cats:
            from_cats = tuple(self._categories.keys())

        cat: str = self._rng.choice(from_cats)
        dataset = self._categories[cat]
        datum: torch.Tensor = dataset[self._rng.integers(len(dataset))]
        return datum

    @abstractmethod
    def __len__(self) -> int:
        """
        Subclasses should decide the artificial multiplier, e.g. based on the
        number of terrain categories. The value returned by the subclass must
        be a multiple of the default value below.
        """
        return self._cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Triplet:
        index = index % self._cumulative_sizes[-1]

        cat_idx: int = np.searchsorted(
            self._cumulative_sizes, index, side="right"
        ).astype(int)

        if cat_idx > 0:
            index -= self._cumulative_sizes[cat_idx - 1]

        # Python spec enforces (iteration order == insertion order)
        cat_names = tuple(self._categories.keys())

        anchor = self._categories[cat_names[cat_idx]][index]
        pos = self._get_random_datum((cat_names[cat_idx],))
        neg = self._get_random_datum((*cat_names[:cat_idx], *cat_names[cat_idx + 1 :]))

        return anchor, pos, neg

    def log_category_sizes(self) -> None:
        for cat, ds in self._categories.items():
            logger.info(f"{cat} data points: {len(ds)}")


class ManualTripletDataset(BaseTripletDataset):
    def __init__(self) -> None:
        super().__init__()

        self._add_category(
            "concrete",
            (
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-16-31-16.bddf",
                    start=1646001080,
                    end=1646001382,
                ),
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-55-35.bddf",
                    start=1646006138,
                    end=1646006272,
                ),
            ),
        )
        self._add_category(
            "grass",
            (
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-16-47-20.bddf",
                    start=1646002045,
                    end=1646002323,
                ),
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-50-10.bddf",
                    start=1646005815,
                    end=1646006071,
                ),
            ),
        )
        self._add_category(
            "sml_rock",
            (
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-24-55.bddf",
                    start=1646004299,
                    end=1646004528,
                ),
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-32-28.bddf",
                    start=1646004753,
                    end=1646005015,
                ),
            ),
        )

        # debug: extend with holdout sets
        self.holdout = ManualTripletHoldoutSet()

    def __len__(self) -> int:
        k = len(self._categories)
        k = k * (k - 1) // 2
        return super().__len__() * k


class ManualTripletHoldoutSet(BaseTripletDataset):
    """Test set to generate embeddings and evaluate model generalizability."""

    def __init__(self) -> None:
        super().__init__()

        self._add_category(
            "concrete",
            (
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-16-59-02.bddf",
                    start=1646607548,
                    end=1646607748,
                ),
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-17-12-30.bddf",
                    start=1646608354,
                    end=1646608594,
                ),
            ),
        )
        self._add_category(
            "grass",
            (
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-16-37-14.bddf",
                    start=1646606240,
                    end=1646606437,
                ),
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-16-48-07.bddf",
                    start=1646606892,
                    end=1646607107,
                ),
            ),
        )
        self._add_category(
            "sml_rock",
            (
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-40-02.bddf",
                    start=1646005206,
                    end=1646005254,
                ),
                SingleTerrainDataset(
                    "data/2022-02-27/2022-02-27-17-41-31.bddf",
                    start=1646005295,
                    end=1646005347,
                ),
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-17-39-58.bddf",
                    start=1646610002,
                    end=1646610098,
                ),
                SingleTerrainDataset(
                    "data/2022-03-06/2022-03-06-17-42-14.bddf",
                    start=1646610138,
                    end=1646610231,
                ),
            ),
        )

    def __len__(self) -> int:
        """This class is not used for training."""
        return 0
