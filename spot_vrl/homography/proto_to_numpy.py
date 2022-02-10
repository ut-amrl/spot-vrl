"""
This module provides helper functions to convert bosdyn.api proto structs to
numpy matrices.
"""

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

import bosdyn.api.geometry_pb2 as geometry_pb2
import bosdyn.api.image_pb2 as image_pb2

from spot_vrl.homography import transform as camera_transform


def se3pose_to_affine(pose: geometry_pb2.SE3Pose) -> npt.NDArray[np.float64]:
    """Converts a SE3Pose proto to a 3D affine transformation matrix.

    Args:
        pose (geometry_pb2.SE3Pose)

    Returns:
        npt.NDArray[np.float64]: The equivalent 4x4 3D affine transformation
            matrix. The return value has the form:

        `[R T]` \n
        `[0 1]`
    """
    q = pose.rotation
    t = pose.position

    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    affine[:3, 3] = [t.x, t.y, t.z]
    return affine


def body_tform_frame(
    tree: geometry_pb2.FrameTreeSnapshot, frame: str
) -> npt.NDArray[np.float64]:
    """Computes the frame-to-body transform from a FrameTreeSnapshot proto.

    The FrameTreeSnapshot is assumed to be rooted at the "body" frame and
    contain the path:
        body -> head -> ... -> frame

    Args:
        tree (geometry_pb2.FrameTreeSnapshot)
        frame (str): The name of the target frame in the tree.

    Returns:
        npt.NDArray[np.float64]: The frame-to-body transform as a 4x4 3D affine
            transformation matrix.
    """
    assert frame in tree.child_to_parent_edge_map
    assert "body" in tree.child_to_parent_edge_map

    root_tform_frame = np.eye(4)
    while frame != "body":
        edge = tree.child_to_parent_edge_map[frame]

        parent_tform_child = se3pose_to_affine(edge.parent_tform_child)
        root_tform_frame = parent_tform_child @ root_tform_frame

        frame = edge.parent_frame_name

    return root_tform_frame


def camera_intrinsic_matrix(
    img_source: image_pb2.ImageSource,
) -> npt.NDArray[np.float64]:
    """Extracts the camera intrinsic matrix from an ImageSource proto.

    Assumes the ImageSource uses a pinhole camera model.

    Args:
        img_source (image_pb2.ImageSource)

    Returns:
        npt.NDArray[np.float64]: The 3x3 camera intrinsic matrix.
    """
    assert img_source.HasField("pinhole")

    intrinsic_proto = img_source.pinhole.intrinsics
    f = intrinsic_proto.focal_length
    p = intrinsic_proto.principal_point
    s = intrinsic_proto.skew

    return np.array([[f.x, s.x, p.x], [s.y, f.y, p.y], [0, 0, 1]])


class SpotImage:
    def __init__(self, image_response: image_pb2.ImageResponse) -> None:
        image_capture = image_response.shot
        image_source = image_response.source
        image = image_capture.image

        assert image_response.status == image_pb2.ImageResponse.Status.STATUS_OK
        assert (
            image_source.image_type == image_pb2.ImageSource.ImageType.IMAGE_TYPE_VISUAL
        )
        assert image.format == image_pb2.Image.Format.FORMAT_JPEG
        assert (
            image.pixel_format == image_pb2.Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8
        )

        self.frame_name = image_capture.frame_name_image_sensor
        self.body_tform_camera = body_tform_frame(
            image_capture.transforms_snapshot, self.frame_name
        )
        self.camera_matrix = camera_intrinsic_matrix(image_source)
        self.width = image.cols
        self.height = image.rows
        self.imgbuf: npt.NDArray[np.uint8] = np.frombuffer(image.data, dtype=np.uint8)

    def decoded_image(self) -> npt.NDArray[np.uint8]:
        """Decodes the raw bytes stored in self.imgbuf as an image.

        Assumes the image is grayscale.

        Returns:
            npt.NDArray[np.uint8]: A 2D matrix of size (self.height, self.width)
                containing a single-channel image.
        """
        img: npt.NDArray[np.uint8] = cv2.imdecode(self.imgbuf, cv2.IMREAD_UNCHANGED)
        assert img.shape == (self.height, self.width)
        return img

    def decoded_image_ground_plane(self) -> npt.NDArray[np.uint8]:
        """Removes the sky from the image returned by self.decoded_image.

        To preserve the size of the image, the color space is converted to RGBA
        and pixels are "removed" by setting them to fully transparent black
        pixels [0, 0, 0, 0].

        Returns:
            npt.NDArray[np.uint8]: A 3D matrix of size (self.height, self.width, 4)
                containing an RGBA image of the visible ground plane. The image is
                still grayscale.
        """
        img = self.decoded_image()
        img: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

        # Implementation note: Python for-loops are pretty slow, so we want to
        #  perform as much computation as possible in numpy at the cost of
        #  space complexity.

        # Generate a 2xN matrix of all integer image coordinates [x, y]
        image_coords = np.indices((self.width, self.height))
        image_coords = np.moveaxis(image_coords, 0, -1)
        image_coords = image_coords.reshape(self.width * self.height, 2)
        image_coords = image_coords.transpose()

        rays = camera_transform.camera_rays(
            image_coords, self.body_tform_camera, self.camera_matrix
        )

        # Find the indices of the rays that point above the horizon
        is_sky: npt.NDArray[np.bool_] = (rays[2] >= 0)

        # Reshape the matrix to avoid computing (x, y) coordinates from flat
        # indices inside a for-loop.
        is_sky = is_sky.reshape(self.width, self.height)
        sky_coords = is_sky.nonzero()

        img[sky_coords[1], sky_coords[0]] = 0

        return img