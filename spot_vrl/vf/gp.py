"""
Visualizes image information for an initial classification proof of concept.

Classifies images from the front cameras as either concrete or grass using
the ground penetration metric.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt

import tqdm

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.api.robot_id_pb2 import RobotIdResponse
from bosdyn.api.robot_state_pb2 import RobotStateResponse, FootState
from bosdyn.bddf import DataReader, ProtobufReader

import spot_vrl.homography
import spot_vrl.homography.transform
from spot_vrl.homography import proto_to_numpy
import spot_vrl.homography.perspective_transform as perspective_transform


# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


class ImageWithText:
    def __init__(self, img: npt.NDArray[np.uint8]) -> None:
        self.img = img
        self.y = 0

    def add_line(
        self, text: str, color: Tuple[int, int, int, int] = (0, 0, 255, 255)
    ) -> None:
        face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.25
        thickness = 3

        text_size = cv2.getTextSize(text, face, scale, thickness)[0]
        self.img = cv2.putText(
            self.img,
            text,
            (0, self.y + text_size[1]),
            face,
            scale,
            color,
            thickness=thickness,
        )
        self.y += text_size[1] + 5


class SensorData:
    """Helper class for storing timestamped sensor data."""

    def __init__(self, filename: str) -> None:

        self.depths: Dict[float, float] = {}
        """Mapping of timestamps to mean ground penetration values."""

        self.images: Dict[float, npt.NDArray[np.uint8]] = {}
        """Mapping of timestamps to fused front camera images."""

        self._datapath: Path = Path(filename)
        self._data_reader = DataReader(None, filename)
        self._proto_reader = ProtobufReader(self._data_reader)

        self._start_ts: float = 0
        """Start timestamp reference for the data file."""

        self._init_start_ts()
        self._init_depths()
        self._init_images()

    def _init_start_ts(self) -> None:
        series_index: int = self._proto_reader.series_index(
            "bosdyn.api.RobotIdResponse"
        )
        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            series_index
        )
        num_msgs = len(series_block_index.block_entries)
        assert num_msgs == 1

        _, ts, _ = self._proto_reader.get_message(series_index, RobotIdResponse, 0)
        self._start_ts = float(ts) * 1e-9

    def _init_depths(self) -> None:
        assert self._start_ts != 0, "Initialize self._start_ts first."

        series_index: int = self._proto_reader.series_index(
            "bosdyn.api.RobotStateResponse"
        )
        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            series_index
        )
        num_msgs = len(series_block_index.block_entries)

        for msg_idx in range(num_msgs):
            _, ts, response = self._proto_reader.get_message(
                series_index, RobotStateResponse, msg_idx
            )
            ts = float(ts) * 1e-9 - self._start_ts
            depth_vals: List[float] = []

            robot_state = response.robot_state
            for foot_state in robot_state.foot_state:
                if foot_state.contact == FootState.Contact.CONTACT_MADE:
                    terrain = foot_state.terrain
                    assert terrain.frame_name == "odom"

                    depth_vals.append(terrain.visual_surface_ground_penetration_mean)

            if depth_vals:
                self.depths[ts] = sum(depth_vals) / len(depth_vals)

    def _init_images(self) -> None:
        assert self._start_ts != 0, "Initialize self._start_ts first."
        assert len(self.depths) != 0, "Initialize self.depths first."

        series_index: int = self._proto_reader.series_index(
            "bosdyn.api.GetImageResponse"
        )
        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            series_index
        )
        num_msgs = len(series_block_index.block_entries)

        ground_tform_body = spot_vrl.homography.transform.affine3d(
            [0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST]
        )

        pbar = tqdm.trange(num_msgs)
        pbar.set_description("Processing Images")
        for msg_idx in pbar:
            _, ts, response = self._proto_reader.get_message(
                series_index, GetImageResponse, msg_idx
            )
            ts = float(ts) * 1e-9 - self._start_ts

            images: List[proto_to_numpy.SpotImage] = []
            for image_response in response.image_responses:
                image = proto_to_numpy.SpotImage(image_response)
                if image.frame_name.startswith("front"):
                    images.append(image)

            fused = perspective_transform.TopDown(images, ground_tform_body)
            self.images[ts] = fused.get_view()

    def save_video(self) -> None:
        save_dir = Path("images") / self._datapath.stem
        frames_dir = save_dir / "frames"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        depth_it = iter(self.depths.items())
        depth_entry = next(depth_it)

        pbar: tqdm.tqdm[int, int] = tqdm.tqdm(self.images.items(), desc="Saving Images")
        for i, (ts, img) in enumerate(pbar):
            img_wrapper = ImageWithText(img)

            img_wrapper.add_line(f"ts: {ts:.3f}s")

            depths: List[float] = []
            while depth_entry[0] < ts:
                depths.append(depth_entry[1])
                try:
                    depth_entry = next(depth_it)
                except StopIteration:
                    depth_entry = (np.inf, 0)

            mean_depth = 0.0
            if depths:
                mean_depth = sum(depths) / len(depths)

            img_wrapper.add_line(f"fd: {mean_depth * 100:.2f} cm")

            category = "concrete" if mean_depth < 0.01 else "grass"
            img_wrapper.add_line(f"cat: {category}")

            # save_path = save_dir / f"top-down-{ts:06.3f}.png"
            save_path = frames_dir / f"{i:04d}.png"
            cv2.imwrite(str(save_path), img_wrapper.img)

        # couldn't get opencv VideoWriter to work on macos
        ffmpeg_args = f"ffmpeg -f image2 -r 8 -i {frames_dir.stem}/%04d.png -c:v libx264 -crf 18 video.mp4".split()
        subprocess.run(ffmpeg_args, cwd=save_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    data = SensorData(options.filename)
    data.save_video()


if __name__ == "__main__":
    main()
