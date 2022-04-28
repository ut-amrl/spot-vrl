import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple, Union, Optional

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.spatial.transform import Rotation
from torch.utils.data import ConcatDataset, Dataset

from spot_vrl.data import ImuData, ImageData
from spot_vrl.homography import camera_transform
from spot_vrl.homography.perspective_transform import TopDown
from spot_vrl.utils.video_writer import VideoWriter

# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters
GROUND_TFORM_BODY = camera_transform.affine3d([0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST])

Patch = torch.Tensor
"""Single channel uint8 image."""

Triplet = Tuple[Patch, Patch, Patch]


class SingleTerrainDataset(Dataset[Tuple[Patch, Patch]]):
    """Image dataset for single trajectories.

    In general, the learning pipeline should generate datasets ahead of time and
    serialize them to disk using the `save()` method.
    """

    output_dir: ClassVar[Path] = Path("visual-datasets")

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
        self.patches: Dict[int, Dict[int, Patch]] = defaultdict(dict)
        """Mapping of image sequence numbers to patches from previous
        viewpoints."""

        imu_data = ImuData(self.path)
        img_data = ImageData(self.path, lazy=False)

        fwd_patches: Dict[int, Dict[int, Tuple[slice, slice]]] = defaultdict(dict)
        """Mapping of image sequence numbers to the future image slices
        generated from this viewpoint. (for video generation)"""

        video_writer: Optional[VideoWriter] = None
        if make_vis_vid:
            video_writer = VideoWriter(self.output_dir / f"{self.path.stem}.mp4")

        for i in tqdm.trange(len(img_data), desc="Loading Dataset"):
            img_ts, images = img_data[i]
            if img_ts < start:
                continue
            elif img_ts > end or img_ts > imu_data.timestamp_sec[-1]:
                break

            imu_ts, poses = imu_data.query_time_range(
                imu_data.tforms("odom", "body"), float(img_ts)
            )
            first_tform_odom = np.linalg.inv(poses[0])

            resolution = 150
            horizon_dist = 4.0
            # The center of the 60x60 patch with this top-left coordinate is
            # (0,0) in the body frame. Using this coordinate as the first patch
            # causes the first few patches in each subtrajectory to contain pixels
            # that are out of view (channel == 0).
            origin_y = 866
            origin_x = 732

            front_images = [img for img in images if "front" in img.frame_name]
            td = TopDown(front_images, GROUND_TFORM_BODY).get_view(
                resolution, horizon_dist
            )

            total_dist = np.float64(0)
            last_odom_pose = poses[0]
            for j in range(i, len(img_data)):
                jmg_ts, _ = img_data[j]

                if jmg_ts > imu_data.timestamp_sec[-1]:
                    break

                _, poses = imu_data.query_time_range(
                    imu_data.tforms("odom", "body"), jmg_ts
                )
                disp = first_tform_odom @ poses[0]

                inst_odom_disp = np.linalg.inv(last_odom_pose) @ poses[0]
                total_dist += np.linalg.norm(inst_odom_disp[:2, 3])
                last_odom_pose = poses[0]

                # The viewable distance is greater than horizon_dist
                # TODO: is this a bug in the TopDown class?
                if total_dist > 6:
                    break

                # Odometry suffers from inaccuracies when turning. Truncate the
                # subtrajectory if the robot turned more than 90 degrees.
                yaw = Rotation.from_matrix(disp[:3, :3]).as_euler("ZXY")[0]
                if abs(yaw) > np.pi / 2:
                    break

                disp_x = disp[0, 3]
                disp_y = disp[1, 3]

                # note: image and world coordinates use different handedness
                tl_x = origin_x - int(disp_y * resolution)
                tl_y = origin_y - int(disp_x * resolution)
                patch_slice = np.s_[tl_y : tl_y + 60, tl_x : tl_x + 60]

                patch = torch.from_numpy(td[patch_slice].copy())

                # Filter out patches that contain out-of-view areas
                if patch.shape == (60, 60) and (patch != 0).all():
                    self.patches[j][i] = patch
                    fwd_patches[i][j] = patch_slice

            if video_writer is not None:
                td = cv2.cvtColor(td, cv2.COLOR_GRAY2BGR)
                for _, s in fwd_patches[i].items():
                    # rectangle() takes x,y point order
                    td = cv2.rectangle(
                        td,
                        (s[1].start, s[0].start),
                        (s[1].stop, s[0].stop),
                        (0, 255, 0),
                    )
                video_writer.add_frame(td)

        if video_writer is not None:
            video_writer.close()

        self.keys = list(self.patches.keys())

    @staticmethod
    def get_serialized_filename(
        bddf_path: Union[str, Path], start: float, end: float
    ) -> str:
        bddf_path = Path(bddf_path)
        return f"{bddf_path.stem}-{start:.0f}-to-{end:.0f}.pkl"

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
        if path.suffix == ".bddf":
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
        patch_ids = tuple(self.patches[viewpoint].keys())

        # Use the clearest possible view of the patch as an anchor
        a_id = max(patch_ids)
        s_id = np.random.choice(patch_ids, size=1)[0]

        # Debug option: use only the clearest and fuzziest patches
        # s_id = min(patch_ids)

        # Completely random sampling
        # a_id, s_id = np.random.choice(patch_ids, size=2, replace=singleton)

        return self.patches[viewpoint][a_id], self.patches[viewpoint][s_id]


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
        with open(filename) as f:
            category: str
            datafiles: List[Dict[str, Any]]
            for category, datafiles in json.load(f)["categories"].items():
                datasets: List[SingleTerrainDataset] = []
                for dataset_spec in datafiles:
                    path: str = dataset_spec["path"]
                    start: float = dataset_spec["start"]
                    end: float = dataset_spec["end"]

                    try:
                        dataset = SingleTerrainDataset.load(path, start, end)
                    except FileNotFoundError:
                        dataset = SingleTerrainDataset(path, start, end, False)
                        dataset.save()

                    datasets.append(dataset)
                self._add_category(category, datasets)
        return self

    def _add_category(self, key: str, datasets: List[SingleTerrainDataset]) -> None:
        self._categories[key] = ConcatDataset(datasets)
        self._cumulative_sizes = np.cumsum(
            [len(ds) for ds in self._categories.values()], dtype=np.int_
        ).tolist()

    def _get_random_datum(self, from_cats: Sequence[str] = ()) -> Tuple[Patch, Patch]:
        """Returns a random (anchor, positive) pair from one of the specified
        categories.

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
        datum: Tuple[Patch, Patch] = dataset[self._rng.integers(len(dataset))]
        return datum

    @abstractmethod
    def __len__(self) -> int:
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

        # anchor, pos = self._categories[cat_names[cat_idx]][index]
        anchor, _ = self._categories[cat_names[cat_idx]][index]
        _, pos = self._get_random_datum((cat_names[cat_idx],))
        neg, _ = self._get_random_datum(
            (*cat_names[:cat_idx], *cat_names[cat_idx + 1 :])
        )

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
        first_t = self.triplet_dataset._get_random_datum((first_cat,))[0]
        second_t = self.triplet_dataset._get_random_datum((second_cat,))[0]

        return first_t, second_t, label
