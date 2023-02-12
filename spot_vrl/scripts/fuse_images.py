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
from spot_vrl.data.sensor_data import SpotSensorData
from spot_vrl.data._deprecated.image_data import ImageData, CameraImage
from spot_vrl.homography._deprecated import camera_transform, perspective_transform
from spot_vrl.utils.video_writer import ImageWithText, VideoWriter
from spot_vrl.utils.parallel import tqdm_position, fork_join


def fuse_images(filename: str) -> None:
    """
    Args:
        filename (str): The path to the BDDF data file.
    """
    filepath = Path(filename)

    spot_data = SpotSensorData(filepath)
    img_data = ImageData.factory(filepath, lazy=True)

    start_ts = img_data[0][0]
    start_tform_odom: npt.NDArray[np.float32] = spot_data.tforms("body", "odom")[0]

    video_writer = VideoWriter(Path("images") / f"{filepath.stem}.mp4", fps=15)

    position = tqdm_position()

    ts: np.float64
    images: List[CameraImage]
    for seq, (ts, images) in tqdm.tqdm(
        enumerate(img_data),
        desc=f"({position}) Processing {filepath}",
        position=position,
        dynamic_ncols=True,
        leave=False,
        total=len(img_data),
    ):
        if ts > spot_data.timestamp_sec[-1]:
            break

        _, odom_poses = spot_data.query_time_range(spot_data.tforms("odom", "body"), ts)
        displacement = (start_tform_odom @ odom_poses[0])[:3, 3]

        _, lin_vels = spot_data.query_time_range(spot_data.linear_vel, ts)
        lin_vel = lin_vels[0]

        td = perspective_transform.TopDown(images)

        # img_wrapper = ImageWithText(cv2.cvtColor(td.get_view(), cv2.COLOR_GRAY2BGR))
        img_wrapper = ImageWithText(td.get_view(resolution=72, horizon_dist=5))
        img_wrapper.add_line(f"seq: {seq}")
        img_wrapper.add_line(f"ts: {ts - start_ts:.3f}")
        img_wrapper.add_line(f"unix: {ts:.3f}")

        img_wrapper.add_line("odom:")
        x, y, z = displacement
        img_wrapper.add_line(f" {x:.2f}")
        img_wrapper.add_line(f" {y:.2f}")
        img_wrapper.add_line(f" {z:.2f}")

        img_wrapper.add_line("v:")
        x, y, z = lin_vel
        img_wrapper.add_line(f" {x:.2f}")
        img_wrapper.add_line(f" {y:.2f}")
        img_wrapper.add_line(f" {z:.2f}")

        fd = spot_data.query_time_range(spot_data.foot_depth_mean, ts)[1][0]
        img_wrapper.add_line(f"depth: {fd}")

        video_writer.add_frame(img_wrapper.img)

    video_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+", help="File to open.")

    options = parser.parse_args()
    # fuse_images(options.filename)

    fork_join(fuse_images, options.filename, 3)


if __name__ == "__main__":
    main()
