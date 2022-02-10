"""
Visualizes image information for an initial classification proof of concept.

Classifies images from the front cameras as either concrete or grass using
the ground penetration metric.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
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

        self._datapath: Path = Path(filename)
        self._data_reader = DataReader(None, filename)
        self._proto_reader = ProtobufReader(self._data_reader)

        self._start_ts: float = 0
        """Start timestamp reference for the data file."""

        self._init_start_ts()
        self._init_depths()

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

    def num_images(self) -> int:
        series_index: int = self._proto_reader.series_index(
            "bosdyn.api.GetImageResponse"
        )
        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            series_index
        )
        return len(series_block_index.block_entries)

    def images(self) -> Iterator[Tuple[float, npt.NDArray[np.uint8]]]:
        """Generates tuples of (timestamp, fused image)."""

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

        for msg_idx in range(num_msgs):
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
            yield ts, fused.get_view()

    def save_video(self) -> None:
        save_dir = Path("images") / self._datapath.stem
        os.makedirs(save_dir, exist_ok=True)

        depth_it = iter(self.depths.items())
        depth_entry = next(depth_it)

        # I couldn't get cv2.VideoWriter to work, so we'll stream to an ffmpeg process

        fps = "8"
        # fmt: off
        ffmpeg_args = [
            "ffmpeg",
            "-loglevel", "warning",
            "-y",
            "-f", "image2pipe",
            "-r", fps,  # need to specify input frame rate, otherwise frames are dropped
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "18",
            "-r", fps,
            "video.mp4",
        ]
        # fmt: on
        ffmpeg = subprocess.Popen(ffmpeg_args, cwd=save_dir, stdin=subprocess.PIPE)
        assert ffmpeg.stdin

        images = self.images()
        pbar: tqdm.tqdm[int, int] = tqdm.tqdm(
            images, desc="Processing Video Frames", total=self.num_images()
        )
        for ts, img in pbar:
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

            img_buf: npt.NDArray[np.uint8]
            _, img_buf = cv2.imencode(".png", img_wrapper.img)
            ffmpeg.stdin.write(img_buf.tobytes())

        ffmpeg.stdin.close()
        ffmpeg.wait()

    def save_plot(self) -> None:
        data_points = len(self.depths)
        timestamps = np.empty(data_points)
        depth_vals = np.empty(data_points)

        for i, (ts, d) in enumerate(self.depths.items()):
            timestamps[i] = ts
            depth_vals[i] = d

        fig = plt.figure(constrained_layout=True, figsize=(12, 7))
        gs = GridSpec(1, 1, figure=fig)

        ax: plt.Axes = fig.add_subplot(gs[0, 0])
        ax.scatter(timestamps, depth_vals)
        ax.set_title("Mean Foot Depth Estimate")
        ax.set_ylabel("meters")
        ax.set_xlabel("seconds")

        save_dir = Path("images") / self._datapath.stem
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / "plot.pdf", format="pdf", dpi=100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    data = SensorData(options.filename)
    data.save_video()
    # data.save_plot()


if __name__ == "__main__":
    main()
