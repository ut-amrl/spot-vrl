"""
This script generates a video visualization of the top-down images generated
from all images in a BDDF
"""

# Disable numpy multithreading
if True:
    import spot_vrl

    spot_vrl._set_omp_num_threads(1)

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import tqdm
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.bddf import DataReader, ProtobufReader
from spot_vrl.data import ImuData, SpotImage
from spot_vrl.homography import camera_transform, perspective_transform
from spot_vrl.utils.video_writer import ImageWithText, VideoWriter

# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


def estimate_fps(proto_reader: ProtobufReader, series_index: int, num_msgs: int) -> int:
    timestamps = []
    for msg_index in range(num_msgs):
        _, ts_nanos, _ = proto_reader.get_message(
            series_index, GetImageResponse, msg_index
        )
        timestamps.append(ts_nanos * 1e-9)

    timestamps = np.array(timestamps, dtype=np.float64)
    period = np.diff(timestamps)
    avg_freq = 1 / period.mean()
    fps = int(np.rint(avg_freq))

    return max(1, fps)


def fuse_images(filename: str) -> None:
    """
    Args:
        filename (str): The path to the BDDF data file.
    """
    filepath = Path(filename)

    data_reader = DataReader(None, filename)
    proto_reader = ProtobufReader(data_reader)
    imu = ImuData(filepath)

    series_index: int = proto_reader.series_index("bosdyn.api.GetImageResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_index)
    num_msgs = len(series_block_index.block_entries)

    ground_tform_body = camera_transform.affine3d([0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST])

    start_ts: Optional[float] = None
    start_tform_odom: npt.NDArray[np.float32] = imu.tforms("body", "odom")[0]

    fps = estimate_fps(proto_reader, series_index, num_msgs)

    video_writer = VideoWriter(Path("images") / f"{filepath.stem}.mp4", fps=fps)
    for msg_index in tqdm.trange(num_msgs, desc="Processing Video"):
        _, ts_nanos, response = proto_reader.get_message(
            series_index, GetImageResponse, msg_index
        )

        ts_sec: float = ts_nanos * 1e-9
        if start_ts is None:
            start_ts = ts_sec

        _, odom_poses = imu.query_time_range(imu.tforms("odom", "body"), ts_sec)
        displacement = np.zeros(3, dtype=np.float32)
        if len(odom_poses) > 0:
            pose = odom_poses[0]
            displacement = (start_tform_odom @ pose)[:3, 3]

        images = []
        for image_response in response.image_responses:
            images.append(SpotImage(image_response))

        td = perspective_transform.TopDown(images, ground_tform_body)

        img_wrapper = ImageWithText(cv2.cvtColor(td.get_view(), cv2.COLOR_GRAY2BGR))
        img_wrapper.add_line(f"seq: {msg_index}")
        img_wrapper.add_line(f"ts: {ts_sec - start_ts:.3f}")
        img_wrapper.add_line(f"unix: {ts_sec:.3f}")

        img_wrapper.add_line("odom:")
        x, y, z = displacement
        img_wrapper.add_line(f" {x:.2f}")
        img_wrapper.add_line(f" {y:.2f}")
        img_wrapper.add_line(f" {z:.2f}")

        video_writer.add_frame(img_wrapper.img)

    video_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    fuse_images(options.filename)


if __name__ == "__main__":
    main()
