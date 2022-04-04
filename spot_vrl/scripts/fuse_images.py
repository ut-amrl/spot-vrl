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
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
import tqdm
from spot_vrl.data import ImuData, SpotImage
from spot_vrl.data.image_data import ImageData
from spot_vrl.homography import camera_transform, perspective_transform
from spot_vrl.utils.video_writer import ImageWithText, VideoWriter

# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


def estimate_fps(img_data: ImageData) -> int:
    timestamps = []
    for ts, _ in img_data:
        timestamps.append(ts)

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

    imu = ImuData(filepath)
    img_data = ImageData(filepath, lazy=True)

    ground_tform_body = camera_transform.affine3d([0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST])
    start_ts = img_data[0][0]
    start_tform_odom: npt.NDArray[np.float32] = imu.tforms("body", "odom")[0]

    fps = estimate_fps(img_data)
    video_writer = VideoWriter(Path("images") / f"{filepath.stem}.mp4", fps=fps)

    ts: np.float64
    images: List[SpotImage]
    for seq, (ts, images) in tqdm.tqdm(
        enumerate(img_data), desc="Processing Video", total=len(img_data)
    ):
        _, odom_poses = imu.query_time_range(imu.tforms("odom", "body"), float(ts))
        displacement = np.zeros(3, dtype=np.float32)
        if len(odom_poses) > 0:
            displacement = (start_tform_odom @ odom_poses[0])[:3, 3]

        td = perspective_transform.TopDown(images, ground_tform_body)

        img_wrapper = ImageWithText(cv2.cvtColor(td.get_view(), cv2.COLOR_GRAY2BGR))
        img_wrapper.add_line(f"seq: {seq}")
        img_wrapper.add_line(f"ts: {ts - start_ts:.3f}")
        img_wrapper.add_line(f"unix: {ts:.3f}")

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
