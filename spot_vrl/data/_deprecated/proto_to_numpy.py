"""
This module provides helper functions to convert bosdyn.api protobuf objects to
numpy structures.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

# from bosdyn.api import geometry_pb2, image_pb2
from google.protobuf.timestamp_pb2 import Timestamp


warnings.warn(
    "The data pipeline for Spot has been fully migrated to ROS."
    " Conversion functions for Boston Dynamics Data Format (.bddf) files"
    " are no longer maintained.",
    category=DeprecationWarning,
    stacklevel=2,
)


def se3pose_to_affine(pose) -> npt.NDArray[np.float64]:
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


def body_tform_frame(tree, frame: str) -> npt.NDArray[np.float64]:
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
    img_source,
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


def timestamp_float64(timestamp: Timestamp) -> np.float64:
    return np.float64(timestamp.seconds) + np.float64(timestamp.nanos) * 1e-9
