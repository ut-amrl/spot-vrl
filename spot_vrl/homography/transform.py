"""This module provides functions to compute homography transforms."""

from typing import Tuple, Literal
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def affine_3d(
    quat: npt.ArrayLike, translation: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Returns a 3D affine transformation matrix.

    [R T]
    [0 1]

    """
    quat = np.asarray(quat)
    translation = np.asarray(translation)

    assert quat.shape == (4,)
    assert translation.shape == (3,)

    tform = np.eye(4)
    tform[:3, :3] = Rotation.from_quat(quat).as_matrix()
    tform[:3, 3] = translation

    return tform


def pixel_direction(
    pixel: npt.ArrayLike,
    world_tform_camera: npt.NDArray[np.float64],
    camera_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Returns the direction vector originating from a camera, expressed in
    world coordinates, that points to a pixel in an image taken by the
    camera.

    The returned vector has a magnitude of 1.

    """
    pixel = np.asarray(pixel, dtype=np.uint)

    assert pixel.shape == (2,)
    assert world_tform_camera.shape == (4, 4)
    assert camera_matrix.shape == (3, 4)

    tform_rot = world_tform_camera[:3, :3]
    vec = np.append(pixel, 1)
    vec /= np.linalg.norm(vec)

    vec: npt.NDArray[np.float64] = tform_rot @ np.linalg.inv(camera_matrix) @ vec

    assert vec.shape == (3,)
    return vec
