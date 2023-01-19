"""
Visualizes image information for an initial classification proof of concept.

Classifies images from the front cameras as either concrete or grass using
the ground penetration metric.
"""

import argparse
import os
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
from bosdyn.bddf import DataReader, ProtobufReader

from spot_vrl.data import ImuData
from spot_vrl.data._deprecated.image_data import SpotImage, CameraImage
from spot_vrl.homography._deprecated import perspective_transform
from spot_vrl.utils.video_writer import VideoWriter


class ImageWithText:
    def __init__(self, img: npt.NDArray[np.uint8]) -> None:
        self.img = img
        self.y = 0

    def add_line(self, text: str, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
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

        imu = ImuData(self._datapath)
        ts = imu.timestamp_sec - self._start_ts

        for t, d in zip(ts, imu.foot_depth_mean):
            self.depths[t] = d

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

        for msg_idx in range(num_msgs):
            _, ts, response = self._proto_reader.get_message(
                series_index, GetImageResponse, msg_idx
            )
            ts = float(ts) * 1e-9 - self._start_ts

            images: List[CameraImage] = []
            for image_response in response.image_responses:
                image = SpotImage(image_response)
                if image.frame_name.startswith("front"):
                    images.append(image)

            fused = perspective_transform.TopDown(images)
            yield ts, fused.get_view()

    def save_video(self) -> None:
        video_writer = VideoWriter(Path("images") / self._datapath.stem / "video.mp4")

        depth_it = iter(self.depths.items())
        depth_entry = next(depth_it)

        images = self.images()
        pbar: tqdm.tqdm[int, int] = tqdm.tqdm(
            images, desc="Processing Video Frames", total=self.num_images()
        )
        for ts, img in pbar:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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

            video_writer.add_frame(img_wrapper.img)
        video_writer.close()

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
