import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, Tuple, Union, Optional

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.data import Dataset

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
            elif img_ts > end:
                break

            imu_ts, poses = imu_data.query_time_range(
                imu_data.tforms("odom", "body"), float(img_ts)
            )
            first_tform_odom = np.linalg.inv(poses[0])

            resolution = 150
            origin_y = 334
            origin_x = 383

            front_images = [img for img in images if "front" in img.frame_name]
            td = TopDown(front_images, GROUND_TFORM_BODY).get_view(resolution)

            patch_slice = np.s_[origin_y : origin_y + 60, origin_x : origin_x + 60]
            self.patches[i][i] = torch.from_numpy(td[patch_slice].copy())
            fwd_patches[i][i] = patch_slice

            for j in range(i, len(img_data)):
                jmg_ts, _ = img_data[j]

                if jmg_ts > imu_data.timestamp_sec[-1]:
                    break

                _, poses = imu_data.query_time_range(
                    imu_data.tforms("odom", "body"), jmg_ts
                )
                disp = first_tform_odom @ poses[0]
                dist = np.linalg.norm(disp[:2, 3])

                disp_x = disp[0, 3]
                disp_y = disp[1, 3]

                # note: image and world coordinates use different handedness
                tl_x = origin_x - int(disp_y * resolution)
                tl_y = origin_y - int(disp_x * resolution)
                patch_slice = np.s_[tl_y : tl_y + 60, tl_x : tl_x + 60]
                self.patches[j][i] = torch.from_numpy(td[patch_slice].copy())
                fwd_patches[i][j] = patch_slice

                if dist > 1.25:
                    break

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

    @classmethod
    def load(cls, path: Union[Path, str]) -> "SingleTerrainDataset":
        """Load a preprocessed dataset from a pickle file.

        Args:
            path (str | Path): Path to the original data file or the
                preprocessed dataset pickle file.

        Raises:
            FileNotFoundError
        """

        path = Path(path)
        if path.suffix == ".bddf":
            path = cls.output_dir / f"{path.stem}.pkl"

        with open(path, "rb") as f:
            obj: SingleTerrainDataset = pickle.load(f)
            if not isinstance(obj, cls):
                logger.warning("Deserialized object does not match type.")
        return obj

    def save(self) -> None:
        """Serialize this dataset to disk."""

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_dir / f"{self.path.stem}.pkl", "wb") as f:
            pickle.dump(self, f, protocol=5)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[Patch, Patch]:
        """Return an anchor and similar patch."""

        viewpoint = self.keys[idx]
        patch_ids = tuple(self.patches[viewpoint].keys())
        singleton = len(patch_ids) == 1
        a_id, s_id = np.random.choice(patch_ids, size=2, replace=singleton)

        return self.patches[viewpoint][a_id], self.patches[viewpoint][s_id]
