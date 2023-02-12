"""
This script generates a video visualization of the images and various stats
contained in a rosbag.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import tqdm
from spot_vrl.data.sensor_data import SpotSensorData
from spot_vrl.data.image_data import Image, KinectImageSequence
from spot_vrl.utils.video_writer import ImageWithText, VideoWriter
from spot_vrl.utils.parallel import tqdm_position, fork_join


def fuse_images(filename: str) -> None:
    """
    Args:
        filename (str): The path to the rosbag.
    """
    filepath = Path(filename)

    spot_data = SpotSensorData(filepath)
    img_data = KinectImageSequence(filepath)

    start_ts = img_data[0][0]
    T_start_odom: npt.NDArray[np.float32] = spot_data.tforms("body", "odom")[0]
    last_image_ts = start_ts - 0.01
    T_odom_lastBody: npt.NDArray[np.float32] = spot_data.tforms("odom", "body")[0]
    odom_distance = np.float32(0)

    video_writer = VideoWriter(Path("images") / f"{filepath.stem}.mp4", fps=15)

    thread_idx = tqdm_position()
    ts: np.float64
    compressed_image: Image
    for seq, (ts, compressed_image) in tqdm.tqdm(
        enumerate(img_data),
        desc=f"({thread_idx}) Processing {filepath}",
        position=thread_idx,
        dynamic_ncols=True,
        leave=False,
        total=len(img_data),
    ):
        if ts > spot_data.timestamp_sec[-1]:
            break

        # extract from return type Tuple[NDArray, NDArray]
        T_odom_currentBody = spot_data.query_time_range(
            spot_data.tforms("odom", "body"), ts
        )[1][0]
        displacement = (T_start_odom @ T_odom_currentBody)[:3, 3]

        # misleading name: "next" is still in the past, but more recent than "lastBody"
        for T_odom_nextBody in spot_data.query_time_range(
            spot_data.tforms("odom", "body"), last_image_ts, ts
        )[1]:
            # where j = i + 1
            T_bodyI_bodyJ = np.linalg.inv(T_odom_lastBody) @ T_odom_nextBody
            # take only norm of xy
            odom_distance += np.linalg.norm(T_bodyI_bodyJ[:2, 3])
            T_odom_lastBody = T_odom_nextBody
        last_image_ts = ts

        lin_vel: npt.NDArray[np.float32] = spot_data.query_time_range(
            spot_data.linear_vel, ts
        )[1][0]

        battery_percent: np.float32 = spot_data.query_time_range(
            spot_data.battery_percent, ts
        )[1][0]

        img_wrapper = ImageWithText(compressed_image.decoded_image())
        img_wrapper.add_line(f"seq: {seq}")
        img_wrapper.add_line(f"ts: {ts - start_ts:.3f}")
        img_wrapper.add_line(f"unix: {ts:.3f}")

        # img_wrapper.add_line("odom:")
        # x, y, z = displacement
        # img_wrapper.add_line(f" {x:.2f}")
        # img_wrapper.add_line(f" {y:.2f}")
        # img_wrapper.add_line(f" {z:.2f}")

        # img_wrapper.add_line("v:")
        # x, y, z = lin_vel
        # img_wrapper.add_line(f" {x:.2f}")
        # img_wrapper.add_line(f" {y:.2f}")
        # img_wrapper.add_line(f" {z:.2f}")

        img_wrapper.add_line(f"odom dist: {odom_distance:.2f} m")

        # ignore z-axis velocity
        img_wrapper.add_line(f"spd: {np.linalg.norm(lin_vel[:2]):.1f} m/s")

        img_wrapper.add_line(f"batt: {battery_percent:.0f}%")

        # fd = spot_data.query_time_range(spot_data.foot_depth_mean, ts)[1][0]
        # img_wrapper.add_line(f"depth: {fd}")

        video_writer.add_frame(img_wrapper.img)

    video_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfiles", type=str, nargs="+", help="Bagfiles to visualize.")

    options = parser.parse_args()

    fork_join(fuse_images, options.bagfiles, 3)


if __name__ == "__main__":
    main()
