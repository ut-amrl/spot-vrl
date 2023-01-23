import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Set, Tuple, Union, Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
import tqdm
from loguru import logger
from scipy.spatial.transform import Rotation
from torch.utils.data import ConcatDataset, Dataset

from spot_vrl.data.sensor_data import SpotSensorData
from spot_vrl.data.image_data import BEVImageSequence
from spot_vrl.utils.video_writer import VideoWriter
import spot_vrl.utils.parallel as parallel


Patch = torch.Tensor
"""uint8 image patch"""

Triplet = Tuple[Patch, Patch, Patch]


def zero_pixel_ratio(image: npt.NDArray[np.uint8]) -> float:
    """Calculate the ratio of zero pixels to all pixels in an image.

    Args:
        image (npt.NDArray[np.uint8]): A HxWx3 image.
    """
    b_or = image[:, :, 0] | image[:, :, 1] | image[:, :, 2]
    return 1.0 - np.count_nonzero(b_or) / (image.shape[0] * image.shape[1])


class SingleTerrainDataset(Dataset[Tuple[Patch, Patch]]):
    """Image dataset for single trajectories.

    Each dataset item is a tuple of (anchor, similar) image patches.

    In general, the learning pipeline should generate datasets ahead of time and
    serialize them to disk using the `save()` method.
    """

    output_dir: ClassVar[Path] = Path("datasets/cache/visual")

    def __init__(
        self,
        path: Union[str, Path],
        start: float = 0,
        end: float = np.inf,
        make_vis_vid: bool = False,
    ) -> None:
        """Constructor for preprocessing.

        Generates an image dataset from scratch from the specified data file.

        Args:
            path (str | Path): Path to the data file.
            start (float): First unix timestamp to start reading from.
            end (float): Last unix timestamp to end reading.
            make_vis_vid (bool): Whether to generate a video visualization of patch
                extraction.
        """

        self.path = Path(path)
        self.start = start
        self.end = end
        self.patch_stack: torch.Tensor
        self.patch_idx_lookup: Dict[int, Dict[int, int]] = defaultdict(dict)
        """Mapping of image sequence numbers to the stack indices of patches
        from previous viewpoints."""
        self.keys: List[int]
        """List of keys in the lookup table."""

        spot_data = SpotSensorData(self.path)
        bev_img_data = BEVImageSequence(self.path)

        fwd_patches: Dict[int, Dict[int, Tuple[slice, slice]]] = defaultdict(dict)
        """Mapping of image sequence numbers to the future image slices
        generated from this viewpoint. (for video generation)"""

        video_writer: Optional[VideoWriter] = None
        if make_vis_vid:
            video_writer = VideoWriter(self.output_dir / f"{self.path.stem}.mp4")

        patches_pre_concat: List[npt.NDArray[np.uint8]] = []
        for i in tqdm.trange(
            len(bev_img_data),
            desc="Loading Dataset",
            position=parallel.tqdm_position(),
            dynamic_ncols=True,
            leave=False,
        ):
            img_ts, image = bev_img_data[i]
            if img_ts < start:
                continue
            elif img_ts > end or img_ts > spot_data.timestamp_sec[-1]:
                break

            spot_ts, poses = spot_data.query_time_range(
                spot_data.tforms("odom", "body"), float(img_ts)
            )
            first_tform_odom = np.linalg.inv(poses[0])

            RESOLUTION = 150  # pixels per meter
            PATCH_SIZE = 64

            bev_image = image.decoded_image()

            # Assume (0, 0) in the body frame is at the center-bottom of this image.
            origin_y = bev_image.shape[0] - PATCH_SIZE // 2
            origin_x = (bev_image.shape[1] // 2) - PATCH_SIZE // 2

            total_dist = np.float64(0)
            last_odom_pose = poses[0]
            for j in range(i, len(bev_img_data)):
                jmg_ts, _ = bev_img_data[j]

                if jmg_ts > spot_data.timestamp_sec[-1]:
                    break

                _, poses = spot_data.query_time_range(
                    spot_data.tforms("odom", "body"), jmg_ts
                )
                disp = first_tform_odom @ poses[0]

                inst_odom_disp = np.linalg.inv(last_odom_pose) @ poses[0]
                total_dist += np.linalg.norm(inst_odom_disp[:2, 3])
                last_odom_pose = poses[0]

                if total_dist > 4:
                    break

                # Odometry suffers from inaccuracies when turning. Truncate the
                # subtrajectory if the robot turned more than 90 degrees.
                yaw = Rotation.from_matrix(disp[:3, :3]).as_euler("ZXY")[0]
                if abs(yaw) > np.pi / 2:
                    break

                disp_x = disp[0, 3]
                disp_y = disp[1, 3]

                # note: image and world coordinates use different handedness
                tl_x = origin_x - int(disp_y * RESOLUTION)
                tl_y = origin_y - int(disp_x * RESOLUTION)
                patch_slice = np.s_[tl_y : tl_y + PATCH_SIZE, tl_x : tl_x + PATCH_SIZE]

                patch = bev_image[patch_slice]

                # Filter out patches that contain out-of-view areas
                if (
                    patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
                    and zero_pixel_ratio(patch) < 0.02
                ):
                    self.patch_idx_lookup[j][i] = len(patches_pre_concat)
                    patches_pre_concat.append(patch)
                    fwd_patches[i][j] = patch_slice

            if video_writer is not None:
                for _, s in fwd_patches[i].items():
                    # rectangle() takes x,y point order
                    bev_image = cv2.rectangle(
                        bev_image,
                        (s[1].start, s[0].start),
                        (s[1].stop, s[0].stop),
                        (0, 255, 0),
                    )
                video_writer.add_frame(bev_image)

        if video_writer is not None:
            video_writer.close()

        stack = np.stack(patches_pre_concat, axis=0)
        stack = np.moveaxis(stack, 3, 1)  # Move channels axis to front for tensors
        self.patch_stack = torch.from_numpy(stack).contiguous()
        logger.debug(f"{self.path.stem}: {self.patch_stack.shape}")
        self.keys = list(self.patch_idx_lookup.keys())

    @staticmethod
    def get_serialized_filename(
        path: Union[str, Path], start: float, end: float
    ) -> str:
        path = Path(path)
        return f"{path.stem}-{start:.0f}-to-{end:.0f}.pkl"

    @classmethod
    def load(
        cls, path: Union[Path, str], start: float, end: float
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

        path = Path(path)
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
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[Patch, Patch]:
        """Returns an anchor and similar patch of the same geolocation as viewed
        from two different viewpoints."""

        viewpoint = self.keys[idx]
        patch_ids = tuple(self.patch_idx_lookup[viewpoint].keys())

        # Use the clearest possible view of the patch as an anchor
        a_idx = max(patch_ids)

        # Use a random view of this patch as a positive
        s_idx = np.random.choice(patch_ids, size=1)[0]

        # Debug option: use only the clearest and fuzziest patches
        # s_idx = min(patch_ids)

        # Completely random sampling
        # singleton = len(patch_ids) == 1
        # a_idx, s_idx = np.random.choice(patch_ids, size=2, replace=singleton)

        return self.patch_stack[a_idx], self.patch_stack[s_idx]


class BaseTripletDataset(Dataset[Triplet], ABC):
    """Base class for Visual Triplet Datasets

    All derived classes must overload `__len__`, optionally using a scaling
    multiplier to artificially increase the number of triplets to generate.

    Example:
    >>> class DerivedTripletDataset(BaseTripletDataset):
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...
    ...     def __len__(self) -> int:
    ...         k = len(self._categories)
    ...         k = k * (k - 1) // 2
    ...         return super().__len__() * k
    """

    def __init__(self) -> None:
        self._categories: Dict[str, ConcatDataset[Tuple[Patch, Patch]]] = {}
        self._cumulative_sizes: List[int] = []
        self._rng = np.random.default_rng()

    @staticmethod
    def _pll_load_or_create(
        category: str, path: str, start: float, end: float
    ) -> Tuple[str, SingleTerrainDataset]:
        try:
            dataset = SingleTerrainDataset.load(path, start, end)
        except FileNotFoundError:
            dataset = SingleTerrainDataset(path, start, end, True)
            dataset.save()
        return category, dataset

    def init_from_json(self, filename: Union[str, Path]) -> "BaseTripletDataset":
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

        categories: Dict[str, List[SingleTerrainDataset]] = defaultdict(list)

        task_args = []
        with open(filename) as f:
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

        for category, dataset in task_results:
            categories[category].append(dataset)

        for category, datasets in categories.items():
            self._add_category(category, datasets)

        return self

    def _add_category(self, key: str, datasets: List[SingleTerrainDataset]) -> None:
        self._categories[key] = ConcatDataset(datasets)
        self._cumulative_sizes = np.cumsum(
            [len(ds) for ds in self._categories.values()], dtype=np.int_
        ).tolist()

    def _get_random_datum(self, category: str) -> Tuple[Patch, Patch]:
        """Returns a random (anchor, positive) pair from the specified category.

        Args:
            category (str): The category to sample from.
        """
        dataset = self._categories[category]
        datum: Tuple[Patch, Patch] = dataset[self._rng.integers(len(dataset))]
        return datum

    @abstractmethod
    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Triplet:
        # completely random selection of two different terrains
        terrain1, terrain2 = random.sample(self._categories.keys(), 2)

        anchor, _ = self._get_random_datum(terrain1)
        _, pos = self._get_random_datum(terrain1)

        # neg, _ = self._get_random_datum(terrain2)
        _, neg = self._get_random_datum(terrain2)

        return anchor, pos, neg

    def log_category_sizes(self) -> None:
        for cat, ds in self._categories.items():
            logger.info(f"{cat} data points: {len(ds)}")


class TripletTrainingDataset(BaseTripletDataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        # k = len(self._categories)
        # k = k * (k - 1) // 2
        # return super().__len__() * k

        # inflate the dataset size based on num categories
        # TODO(eyang): why? is optimization not stable when this
        # is too low
        size = len(self._categories) * super().__len__()

        # deflate the dataset size if it's really large
        # so that training isn't extremely slow
        return min(size, 100_000)


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
        first_t = self.triplet_dataset._get_random_datum(first_cat)[0]
        second_t = self.triplet_dataset._get_random_datum(second_cat)[0]

        return first_t, second_t, label
