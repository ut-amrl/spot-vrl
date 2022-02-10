"""This module provides functions to compute homography transforms."""

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def affine3d(
    quat: npt.ArrayLike, translation: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Constructs a 3D affine transformation matrix.

    Args:
        quat (npt.ArrayLike): A XYZW-order quaternion.
        translation (npt.ArrayLike): A 3D vector.

    Returns:
        npt.NDArray[np.float64]: The equivalent 4x4 3D affine transformation
            matrix.
    """
    quat = np.asarray(quat)
    translation = np.asarray(translation)

    assert quat.shape == (4,)
    assert translation.shape == (3,)

    tform = np.eye(4)
    tform[:3, :3] = Rotation.from_quat(quat).as_matrix()
    tform[:3, 3] = translation

    return tform


def to_homogenous(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts regular coordinates into homogeneous coordinates.

    Args:
        coords (npt.NDArray[np.float64]): A matrix of column vectors.

    Returns:
        npt.NDArray[np.float64]: The input matrix with an additional
            row of ones.
    """
    if coords.ndim == 1:
        coords = coords.reshape(coords.shape[0], 1)

    assert coords.ndim == 2
    _, cols = coords.shape
    return np.vstack((coords, np.ones(cols, dtype=np.float64)))


def from_homogeneous(coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Converts homogeneous coordinates to regular coordinates.

    Args:
        coords (npt.NDArray[np.float64]): A matrix of column vectors.

    Returns:
        npt.NDArray[np.float64]: The input matrix without its last row.
    """
    if coords.ndim == 1:
        coords = coords.reshape(coords.shape[0], 1)

    assert coords.ndim == 2
    return coords[:-1]


def camera_rays(
    image_coords: npt.ArrayLike,
    world_tform_camera: npt.NDArray[np.float64],
    camera_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates the rays originating from a camera in the world frame that
    point to pixels in the camera's image.

    Args:
        image_coords (npt.ArrayLike): A 2xN matrix of (x, y) image coordinates.
        world_tform_camera (npt.NDArray[np.float64]): The camera-to-world affine
            transform matrix.
        camera_matrix (npt.NDArray[np.float64]): The camera intrinsic matrix.

    Returns:
        npt.NDArray[np.float64]: A 3xN matrix of rays originating from the
            camera frame. The rays have arbitrary magnitude.
    """
    image_coords = np.asarray(image_coords, dtype=np.float64)
    assert image_coords.shape[0] == 2

    tform_rot = world_tform_camera[:3, :3]
    rays = tform_rot @ np.linalg.inv(camera_matrix) @ to_homogenous(image_coords)

    return rays


def image_to_ground(
    image_coords: npt.NDArray[np.float64],
    ground_tform_camera: npt.NDArray[np.float64],
    camera_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates the ground plane coordinates corresponding to image
    coordinates.

    Args:
        image_coords (npt.NDArray[np.float64]): A 2xN matrix of (x, y) image
            coordinates.
        ground_tform_camera (npt.NDArray[np.float64]): The camera-to-ground-plane
            affine transform matrix.
        camera_matrix (npt.NDArray[np.float64]): The camera intrinsic matrix.

    Returns:
        npt.NDArray[np.float64]: A 3xN matrix of ground plane coordinates. The z
            value of each coordinate is zero, unless the corresponding image
            pixel was not on the ground plane (i.e. at or above the horizon), in
            which case the entire column is set to NaN.
    """
    assert image_coords.shape[0] == 2

    if image_coords.ndim == 1:
        n_points = 1
    else:
        assert image_coords.ndim == 2
        n_points = image_coords.shape[1]

    camera_loc = ground_tform_camera[:3, 3]
    rays = camera_rays(image_coords, ground_tform_camera, camera_matrix)

    # extend camera rays to the ground plane (z=0)
    for i in range(n_points):
        ray = rays[:, i]

        if ray[2] >= 0:
            rays[:, i] = np.NaN
        else:
            k = -camera_loc[2] / ray[2]
            rays[:, i] = camera_loc + k * ray

    return rays


def ground_to_image(
    ground_coords: npt.NDArray[np.float64],
    camera_tform_ground: npt.NDArray[np.float64],
    camera_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculates the projected image coordinates of ground plane coordinates.

    Args:
        ground_coords (npt.NDArray[np.float64]): A 3xN matrix of ground plane
            coordinates.
        camera_tform_ground (npt.NDArray[np.float64]): The ground-to-camera
            affine transform matrix.
        camera_matrix (npt.NDArray[np.float64]): The camera intrinsic matrix.

    Returns:
        npt.NDArray[np.float64]: A 2xN matrix of (x, y) image coordinates.
    """
    assert ground_coords.shape[0] == 3

    camera_coords = from_homogeneous(camera_tform_ground @ to_homogenous(ground_coords))
    projected_coords = camera_matrix @ camera_coords
    projected_coords /= projected_coords[2]
    return projected_coords[:2]


def visible_ground_plane_limits(
    ground_tform_camera: npt.NDArray[np.float64],
    camera_matrix: npt.NDArray[np.float64],
    img_width: int,
    img_height: int,
    horizon_dist: float = 2,
) -> npt.NDArray[np.float64]:
    """Estimates the visible limits of the ground plane as viewed from a camera.

    Assumes some set of two adjacent corners in the camera image contain the
    ground plane and the two opposite corners do not contain the ground plane.

    The two remaining ground plane coordinates are calculated by estimating
    the horizon.

    Args:
        ground_tform_camera (npt.NDArray[np.float64]): The camera-to-ground
            affine transform matrix.
        camera_matrix (npt.NDArray[np.float64]): The camera intrinsic matrix.
        img_width (int)
        img_height (int)
        horizon_dist (float): The distance from the camera to the horizon.

    Returns:
        npt.NDArray[np.float64]: A 3x4 matrix of ground plane coordinates.
    """
    ground_plane_limits = np.empty((3, 0))

    # fmt: off
    image_limits = np.array([
        [0, img_width - 1,              0, img_width - 1],
        [0,             0, img_height - 1, img_height - 1]
    ])
    # fmt: on
    image_limits_in_plane = image_to_ground(
        image_limits, ground_tform_camera, camera_matrix
    )

    for col in image_limits_in_plane.transpose():
        col = col.reshape((3, 1))
        if not np.isnan(col).any():
            ground_plane_limits = np.hstack((ground_plane_limits, col))
    assert ground_plane_limits.shape == (3, 2)

    # Estimate the endpoints of the horizon in the image by fixing a point
    # `horizon_dist` directly in front of the camera and finding where the
    # orthogonal vectors intersect with the boundary of the image.
    #
    # This approach is invariant to the orientation of the camera.

    # Project the camera's z axis and location onto the ground plane.
    ground_rot_camera = ground_tform_camera[:3, :3]
    camera_dir = np.array([0, 0, 1])
    camera_dir = ground_rot_camera @ camera_dir
    camera_dir[2] = 0
    camera_dir /= np.linalg.norm(camera_dir)

    camera_loc = np.copy(ground_tform_camera[:3, 3])
    camera_loc[2] = 0

    horizon_point = camera_loc + horizon_dist * camera_dir
    orthogonal = np.array([-camera_dir[1], camera_dir[0], 0])

    def out_of_image(x: float, y: float) -> bool:
        return x < 0 or x > img_width - 1 or y < 0 or y > img_height - 1

    camera_tform_ground = np.linalg.inv(ground_tform_camera)
    for _ in range(2):
        # binary search for the ground plane distance to the image boundary
        # intersection
        EPSILON = 1e-4
        low = 0.0
        high = 999.0

        while (high - low > EPSILON):
            mid = (low + high) * 0.5

            ground_coord = horizon_point + mid * orthogonal
            image_coord = ground_to_image(ground_coord, camera_tform_ground, camera_matrix)
            if (out_of_image(image_coord[0], image_coord[1])):
                high = mid - EPSILON
            else:
                low = mid

        intersection = (horizon_point + low * orthogonal).reshape((3, 1))
        ground_plane_limits = np.hstack((ground_plane_limits, intersection))

        orthogonal = -orthogonal

    assert ground_plane_limits.shape == (3, 4)
    return ground_plane_limits
