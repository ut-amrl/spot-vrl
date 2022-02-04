"""
This module provides helper functions to convert bosdyn.api proto structs to
numpy matrices.
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

import bosdyn.api.geometry_pb2 as geometry_pb2
import bosdyn.api.image_pb2 as image_pb2


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
