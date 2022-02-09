"""
This script uses homography transforms to fuse the images from a BDDF data file
into top-down images.
"""

import argparse
from typing import List
from pathlib import Path

import bosdyn.bddf
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import (
    GetImageResponse,
)
from bosdyn.api.robot_id_pb2 import RobotIdResponse

from spot_vrl.homography import proto_to_numpy
import spot_vrl.homography.transform  # noqa: F401
import spot_vrl.homography.perspective_transform as perspective_transform


# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


def get_ref_ts(filename: str) -> float:
    data_reader = bosdyn.bddf.DataReader(None, filename)
    proto_reader = bosdyn.bddf.ProtobufReader(data_reader)

    series_index: int = proto_reader.series_index("bosdyn.api.RobotIdResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_index)
    num_msgs = len(series_block_index.block_entries)
    assert num_msgs == 1

    _, ts, _ = proto_reader.get_message(series_index, RobotIdResponse, 0)
    return float(ts) * 1e-9


def fuse_images(filename: str) -> None:
    data_reader = bosdyn.bddf.DataReader(None, filename)
    proto_reader = bosdyn.bddf.ProtobufReader(data_reader)

    series_index: int = proto_reader.series_index("bosdyn.api.GetImageResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_index)
    num_msgs = len(series_block_index.block_entries)

    ground_tform_body = spot_vrl.homography.transform.affine3d(
        [0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST]
    )

    start_ts = get_ref_ts(filename)

    for msg_index in range(num_msgs):
        _, ts, response = proto_reader.get_message(
            series_index, GetImageResponse, msg_index
        )

        ts = float(ts) * 1e-9 - start_ts

        images: List[proto_to_numpy.SpotImage] = []

        for image_response in response.image_responses:
            images.append(proto_to_numpy.SpotImage(image_response))

        td = perspective_transform.TopDown(images, ground_tform_body)

        save_dir = Path("images") / Path(filename).stem
        td.save(save_dir / f"top-down-{ts:.4f}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    fuse_images(options.filename)


if __name__ == "__main__":
    main()
