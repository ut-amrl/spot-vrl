import os
from pathlib import Path
from typing import Union, List, Dict

import numpy as np
import numpy.typing as npt
import cv2

from spot_vrl.data import SpotImage
from spot_vrl.homography import camera_transform


class TopDown:
    def __init__(
        self, spot_images: List[SpotImage], ground_tform_body: npt.NDArray[np.float64]
    ) -> None:
        self.spot_images = spot_images
        self.ground_tform_body = ground_tform_body

    def get_view(
        self, resolution: int = 100, horizon_dist: float = 2.0
    ) -> npt.NDArray[np.uint8]:
        """Generate a top-down view of the ground around the robot.

        The image is oriented such that:
        - The robot's X axis points upwards from the center of the image.
        - The robot's Y axis points left from the center of the image.

        Args:
            resolution (int): The output resolution in pixels per meter of
                ground.

        Returns:
            npt.NDArray[np.uint8]: A 2D matrix of containing a single-channel
                image of the ground around the robot. Pixels with value 0 are
                invalid (e.g. out of view).
        """
        all_plane_limits: Dict[str, npt.NDArray[np.float64]] = {}
        for spot_image in self.spot_images:
            ground_tform_camera = self.ground_tform_body @ spot_image.body_tform_camera
            all_plane_limits[
                spot_image.frame_name
            ] = camera_transform.visible_ground_plane_limits(
                ground_tform_camera,
                spot_image.camera_matrix,
                spot_image.width,
                spot_image.height,
                horizon_dist=horizon_dist,
            )

        view_limits = np.zeros((2, 2))  # min, max as xy row vectors
        for limits in all_plane_limits.values():
            for col in limits.transpose():
                view_limits[0] = np.minimum(view_limits[0], col[:2])
                view_limits[1] = np.maximum(view_limits[1], col[:2])

        output_width = int(np.ceil((view_limits[1] - view_limits[0])[1] * resolution))
        output_height = int(np.ceil((view_limits[1] - view_limits[0])[0] * resolution))

        output_img: npt.NDArray[np.uint8] = np.zeros(
            (output_height, output_width), dtype=np.uint8
        )

        for spot_image in self.spot_images:
            ground_tform_camera = self.ground_tform_body @ spot_image.body_tform_camera
            camera_tform_ground = np.linalg.inv(ground_tform_camera)

            plane_limits = all_plane_limits[spot_image.frame_name]

            # Calculate the source image coordinates of this camera's ground
            # plane limits.
            limits_source_coords = camera_transform.ground_to_image(
                plane_limits, camera_tform_ground, spot_image.camera_matrix
            )

            # Calculate the output image coordinates of this camera's ground
            # plane limits.
            limits_output_coords = np.empty((2, 4))
            limits_output_coords[0] = (view_limits[1, 1] - plane_limits[1]) * resolution
            limits_output_coords[1] = (view_limits[1, 0] - plane_limits[0]) * resolution

            perspective_transform = cv2.getPerspectiveTransform(
                limits_source_coords.astype(np.float32).T,
                limits_output_coords.astype(np.float32).T,
            )

            ground_img = spot_image.decoded_image_ground_plane()
            warped_ground_img: npt.NDArray[np.uint8] = cv2.warpPerspective(
                ground_img, perspective_transform, (output_width, output_height)
            )

            # Fill in empty pixels in the output image.
            output_img = np.where(
                output_img == 0,
                warped_ground_img,
                output_img,
            )

        return output_img

    def save(self, filename: Union[Path, str]) -> None:
        filename = Path(filename)
        if not os.path.isdir(filename.parent):
            os.makedirs(filename.parent)

        cv2.imwrite(str(filename), self.get_view())
