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
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot

from spot_vrl import homography as homography
import spot_vrl.homography.transform  # noqa: F401
from scipy.spatial.transform import Rotation


def get_body_tform_camera(
    tree: FrameTreeSnapshot, camera_frame: str
) -> npt.NDArray[np.float64]:
    assert camera_frame in tree.child_to_parent_edge_map
    assert "body" in tree.child_to_parent_edge_map

    root_tform_camera = np.eye(4)

    # move root of subtree up until it is the body frame
    #
    # assumes body is the true root of the tree
    # body -> head -> ... -> camera_frame
    # body and head should be identity transforms
    frame = camera_frame
    while frame in tree.child_to_parent_edge_map:
        edge = tree.child_to_parent_edge_map[frame]
        q = edge.parent_tform_child.rotation
        tx = edge.parent_tform_child.position

        parent_tform_child = homography.transform.affine_3d(
            [q.x, q.y, q.z, q.w], [tx.x, tx.y, tx.z]
        )

        root_tform_camera = parent_tform_child @ root_tform_camera

        frame = edge.parent_frame_name

    return root_tform_camera


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

            body_tform_camera = get_body_tform_camera(
                img_capture.transforms_snapshot, img_capture.frame_name_image_sensor
            )
            rotmat = body_tform_camera[:3, :3]
            print(Rotation.from_matrix(rotmat).as_quat())

            translation = body_tform_camera[:3, 3]
            print(translation)
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    fuse_images(options.filename)


if __name__ == "__main__":
    main()
