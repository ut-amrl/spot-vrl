"""
Visualizes trajectory and IMU data to aid with time window selection.

In general, windows can be filtered/selected based on:
  - visual terrain type
  - ground speed > threshold
  - ground depth > 0 and matches terrain expectations
"""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
import tqdm

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.bddf import DataReader, ProtobufReader
from spot_vrl import homography
from spot_vrl.data import ImuData
from spot_vrl.homography.perspective_transform import TopDown
from spot_vrl.homography.proto_to_numpy import SpotImage
from spot_vrl.scripts.gp import ImageWithText  # maybe a bad dependency
from spot_vrl.utils.video_writer import VideoWriter


class ImageIterator:
    def __init__(self, path: Path) -> None:
        self._data_reader = DataReader(None, str(path))
        self._proto_reader = ProtobufReader(self._data_reader)

        self._series_index: int = self._proto_reader.series_index(
            "bosdyn.api.GetImageResponse"
        )

        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            self._series_index
        )
        self.num_msgs = len(series_block_index.block_entries)
        self._iter_idx = 0

        self._ground_tform_body = homography.transform.affine3d(
            [0, 0, 0, 1], [0, 0, 0.48938]
        )

    def __iter__(self) -> "ImageIterator":
        self._iter_idx = 0
        return self

    def __next__(self) -> Tuple[float, npt.NDArray[np.uint8]]:
        if self._iter_idx >= self.num_msgs:
            raise StopIteration

        _, ts, response = self._proto_reader.get_message(
            self._series_index, GetImageResponse, self._iter_idx
        )

        images = []
        for image_response in response.image_responses:
            image = SpotImage(image_response)
            if image.frame_name.startswith("front"):
                images.append(image)

        fused = TopDown(images, self._ground_tform_body)

        self._iter_idx += 1
        return float(ts) * 1e-9, fused.get_view(resolution=150)


def vis_data(path: Path) -> None:
    imu = ImuData(path)
    images = ImageIterator(path)
    vid = VideoWriter(Path("images") / path.stem / "imu_vis.mp4")

    for _ in tqdm.trange(images.num_msgs, desc="Processing Video"):
        ts, image = next(images)

        _, linear_vels = imu.query_time_range(imu.linear_vel, start=ts, end=ts + 1)
        _, foot_depths = imu.query_time_range(imu.foot_depth_mean, start=ts, end=ts + 1)

        if linear_vels.size == 0:
            continue

        if linear_vels.ndim > 1:
            linear_vels = linear_vels[0]

        linear_spd = np.linalg.norm(linear_vels)
        foot_depth = foot_depths[0]

        img_wrapper = ImageWithText(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        img_wrapper.add_line(f"ts: {ts:.3f}")
        img_wrapper.add_line(f"fd: {foot_depth * 100:.2f} cm")
        img_wrapper.add_line(f"spd: {linear_spd:.3f} m/s")

        vid.add_frame(img_wrapper.img)
    vid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=Path)

    args = parser.parse_args()
    vis_data(args.filename)
