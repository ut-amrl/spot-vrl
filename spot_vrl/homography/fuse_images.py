"""
This script uses homography transforms to fuse the images from a BDDF data file
into top-down images.
"""

import argparse

import numpy as np
import numpy.typing as npt

import bosdyn.bddf
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import (
    GetImageResponse,
    Image,
    ImageResponse,
    ImageSource,
)

from spot_vrl.homography import proto_to_numpy

from scipy.spatial.transform import Rotation


def fuse_images(filename: str) -> None:
    data_reader = bosdyn.bddf.DataReader(None, filename)
    proto_reader = bosdyn.bddf.ProtobufReader(data_reader)

    series_index: int = proto_reader.series_index("bosdyn.api.GetImageResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_index)
    num_msgs = len(series_block_index.block_entries)

    for msg_index in range(1):
        _, _, response = proto_reader.get_message(
            series_index, GetImageResponse, msg_index
        )

        for image_response in response.image_responses:
            assert image_response.status == ImageResponse.Status.STATUS_OK
            img_capture = image_response.shot
            print(img_capture.frame_name_image_sensor)

            image = img_capture.image
            assert image.format == Image.Format.FORMAT_JPEG
            assert image.pixel_format == Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8

            img_source = image_response.source
            assert img_source.image_type == ImageSource.ImageType.IMAGE_TYPE_VISUAL

            body_tform_camera = proto_to_numpy.body_tform_frame(
                img_capture.transforms_snapshot, img_capture.frame_name_image_sensor
            )
            rotmat = body_tform_camera[:3, :3]
            print(Rotation.from_matrix(rotmat).as_quat())

            translation = body_tform_camera[:3, 3]
            print(translation)

            camera_matrix = proto_to_numpy.camera_intrinsic_matrix(img_source)
            print(camera_matrix)

            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    fuse_images(options.filename)


if __name__ == "__main__":
    main()
