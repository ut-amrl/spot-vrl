"""
This script generates a video visualization of the top-down images generated
from all images in a BDDF
"""

import argparse
from pathlib import Path

import tqdm
import bosdyn.bddf
import spot_vrl.homography.perspective_transform as perspective_transform
import spot_vrl.homography.transform
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from spot_vrl.homography import proto_to_numpy
from spot_vrl.utils.video_writer import VideoWriter

# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


def fuse_images(filename: str) -> None:
    """
    Args:
        filename (str): The path to the BDDF data file.
    """
    filepath = Path(filename)
    video_writer = VideoWriter(Path("images") / f"{filepath.stem}.mp4")

    data_reader = bosdyn.bddf.DataReader(None, filename)
    proto_reader = bosdyn.bddf.ProtobufReader(data_reader)

    series_index: int = proto_reader.series_index("bosdyn.api.GetImageResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_index)
    num_msgs = len(series_block_index.block_entries)

    ground_tform_body = spot_vrl.homography.transform.affine3d(
        [0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST]
    )

    for msg_index in tqdm.trange(num_msgs, desc="Processing Video"):
        _, _, response = proto_reader.get_message(
            series_index, GetImageResponse, msg_index
        )

        images = []
        for image_response in response.image_responses:
            images.append(proto_to_numpy.SpotImage(image_response))

        td = perspective_transform.TopDown(images, ground_tform_body)
        video_writer.add_frame(td.get_view())

    video_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    fuse_images(options.filename)


if __name__ == "__main__":
    main()
