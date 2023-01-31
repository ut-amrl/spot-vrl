import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import torch
import torch.utils.data
import tqdm
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset

from spot_vrl.data.sensor_data import SpotSensorData
import spot_vrl.utils.parallel as parallel


class SingleTerrainDataset(Dataset[torch.Tensor]):
    """Sensor dataset from a single terrain type.

    Data are time windows as fixed-sized matrices in the layout:

          *-------> data types
          |
          |
          |
          V
        time
    """

    window_size: ClassVar[float] = 3.0
    """seconds"""

    output_dir: ClassVar[Path] = Path("datasets/cache/inertial")

    @classmethod
    def set_global_window_size(cls, new_size: int) -> None:
        """Sets the window size to be used by future instances of this class.

        Has no effect on existing instances.
        """
        if new_size < 1:
            raise ValueError("Window size must be positive")
        cls.window_size = new_size

    def __init__(
        self,
        path: Union[str, Path],
        start: float = 0,
        end: float = np.inf,
    ) -> None:
        self.path = Path(path)
        self.start = start
        self.end = end

        spot_data = SpotSensorData(path)

        window_size = self.window_size
        self.windows: torch.Tensor

        windows_pre_concat: List[npt.NDArray[np.float32]] = []
        ts, _ = spot_data.query_time_range(spot_data.all_sensor_data, start, end)
        # Overlap windows for more data points
        window_start: float
        for window_start in tqdm.tqdm(
            np.arange(ts[0], ts[-1] - window_size, 0.5),
            desc="Loading Dataset",
            position=parallel.tqdm_position(),
            dynamic_ncols=True,
            leave=False,
        ):
            _, window = spot_data.query_time_range(
                spot_data.all_sensor_data[:, -4:],
                window_start,
                window_start + window_size,
            )

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
            window = np.vstack((mean, std, skew, kurtosis, med, q1, q3)).astype(
                np.float32
            )
            windows_pre_concat.append(window)

        stack = np.stack(windows_pre_concat, axis=0)
        self.windows = torch.from_numpy(stack).contiguous()

    @staticmethod
    def get_serialized_filename(
        path: Union[str, Path], start: float, end: float
    ) -> str:
        path = Path(path)
        return f"{path.stem}-{start:.0f}-to-{end:.0f}.pkl"

    @classmethod
    def load(
        cls, path: Union[str, Path], start: float, end: float
    ) -> "SingleTerrainDataset":
        """Load a preprocessed dataset from a pickle file.

        Args:
            path (str | Path): Path to the original data file or the
                preprocessed dataset pickle file.
            start (float): First unix timestamp to start reading from.
            end (float): Last unix timestamp to end reading.

        Raises:
            FileNotFoundError
        """
        path = cls.output_dir / cls.get_serialized_filename(path, start, end)

        with open(path, "rb") as f:
            obj: SingleTerrainDataset = pickle.load(f)
            if not isinstance(obj, cls):
                logger.warning("Deserialized object does not match type.")
        return obj

    def save(self) -> None:
        """Serialize this dataset to disk."""

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = self.output_dir / self.get_serialized_filename(
            self.path, self.start, self.end
        )
        with open(save_path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


Triplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BaseTripletDataset(Dataset[Triplet]):
    """Base class for Sensor Triplet Datasets

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

    @staticmethod
    def _pll_load_or_create(
        category: str, path: str, start: float, end: float
    ) -> Tuple[str, SingleTerrainDataset]:
        try:
            dataset = SingleTerrainDataset.load(path, start, end)
        except FileNotFoundError:
            dataset = SingleTerrainDataset(path, start, end)
            dataset.save()
        return category, dataset

    def init_from_json(self, spec_filename: Union[str, Path]) -> "BaseTripletDataset":
        """Initializes/overwrites this dataset using a JSON specification file.

        The JSON file is expected to contain the following format:

        {
            "categories": {
                "name-of-terrain-1": [
                    {
                        "path": "relative/to/project/root",
                        "start": 1646607548,
                        "end": 1646607748
                    },
                    ...
                ],
                ...
            }
        }
        """
        task_args = []
        with open(spec_filename) as f:
            category: str
            datafiles: List[Dict[str, Any]]
            for category, datafiles in json.load(f)["categories"].items():
                for dataset_spec in datafiles:
                    path: str = dataset_spec["path"]
                    start: float = dataset_spec["start"]
                    end: float = dataset_spec["end"]
                    task_args.append((category, path, start, end))

        task_results = parallel.fork_join(
            BaseTripletDataset._pll_load_or_create, task_args, n_proc=8
        )

        categories: Dict[str, List[SingleTerrainDataset]] = defaultdict(list)
        for category, dataset in task_results:
            categories[category].append(dataset)

        for category, datasets in categories.items():
            self._add_category(category, datasets)

        return self

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


class TripletTrainingDataset(BaseTripletDataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        k = len(self._categories)
        k = k * (k - 1) // 2
        return super().__len__() * k


class TripletHoldoutDataset(BaseTripletDataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        """Prevent this class from being used for training."""
        return 0


class PairCostTrainingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, float]]):
    def __init__(self, spec: Union[str, Path]) -> None:
        self.triplet_dataset = TripletTrainingDataset()
        self.orderings: Dict[Tuple[str, str], float] = {}

        self.triplet_dataset.init_from_json(spec)

        with open(spec) as f:
            seen_categories: Set[str] = set()

            # TODO: check for transitivity
            for order in json.load(f)["orderings"]:
                first: str = order["first"]
                second: str = order["second"]
                label: float = float(order["label"])

                if first not in self.triplet_dataset._categories:
                    logger.error(f"({spec}): {first} not in categories set")
                    raise ValueError

                if second not in self.triplet_dataset._categories:
                    logger.error(f"({spec}): {second} not in categories set")
                    raise ValueError

                if first == second:
                    logger.error(f"({spec}): first==second")
                    raise ValueError

                if label not in (1.0, -1.0, 0.0):
                    logger.error(f"({spec}): label number {label} not in (-1, 0, or 1)")
                    raise ValueError

                if self.orderings.setdefault((first, second), label) != label:
                    logger.error(f"({spec}): ordering mismatch for ({first}, {second})")
                    raise ValueError

                if self.orderings.get((second, first), -label) != -label:
                    logger.error(f"({spec}): ordering mismatch for ({second}, {first})")
                    raise ValueError

                seen_categories.add(first)
                seen_categories.add(second)

            for category in self.triplet_dataset._categories.keys():
                self.orderings[(category, category)] = 0.0

            if seen_categories != set(self.triplet_dataset._categories.keys()):
                logger.error(f"({spec}): orderings do not contain all categories")

    def __len__(self) -> int:
        return len(self.triplet_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        (first_cat, second_cat), label = random.choice(tuple(self.orderings.items()))
        first_t = self.triplet_dataset._get_random_datum((first_cat,))
        second_t = self.triplet_dataset._get_random_datum((second_cat,))

        return first_t, second_t, label
