from typing import List, Dict

import numpy as np
import numpy.typing as npt
import cv2

from spot_vrl.homography.proto_to_numpy import SpotImage
from spot_vrl.homography import transform as camera_transform


class TopDown:
    # transparent pixel in RGBA
    ZERO_PIXEL = np.array([0, 0, 0, 0])

    def __init__(
        self, spot_images: List[SpotImage], ground_tform_body: npt.NDArray[np.float64]
    ) -> None:
        self.spot_images = spot_images
        self.ground_tform_body = ground_tform_body

    def get_view(self) -> npt.NDArray[np.uint8]:
        """Generate a top-down view of the ground around the robot.

        Returns:
            npt.NDArray[np.uint8]: A 3D matrix of containing an RGBA image of
                the ground around the robot.
        """
        OUTPUT_RESOLUTION = 100  # pixels per meter

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
            )

        view_limits = np.zeros((2, 2))  # min, max as xy row vectors
        for limits in all_plane_limits.values():
            for col in limits.transpose():
                view_limits[0] = np.minimum(view_limits[0], col[:2])
                view_limits[1] = np.maximum(view_limits[1], col[:2])

        print("view limits: [x y]")
        print(f"\tmin: {view_limits[0]}")
        print(f"\tmax: {view_limits[1]}")

        output_width = int(
            np.ceil((view_limits[1] - view_limits[0])[0] * OUTPUT_RESOLUTION)
        )
        output_height = int(
            np.ceil((view_limits[1] - view_limits[0])[1] * OUTPUT_RESOLUTION)
        )

        print(f"output img dim: {output_width} x {output_height}")
        output_img: npt.NDArray[np.uint8] = np.zeros(
            (output_height, output_width, 4), dtype=np.uint8
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
            limits_output_coords[0] = (
                -(plane_limits[0] - view_limits[1, 0]) * OUTPUT_RESOLUTION
            )
            limits_output_coords[1] = (
                plane_limits[1] - view_limits[0, 1]
            ) * OUTPUT_RESOLUTION

            perspective_transform = cv2.getPerspectiveTransform(
                limits_source_coords.astype(np.float32).T,
                limits_output_coords.astype(np.float32).T,
            )

            ground_img = spot_image.decoded_image_ground_plane()
            warped_ground_img: npt.NDArray[np.uint8] = cv2.warpPerspective(
                ground_img, perspective_transform, (output_width, output_height)
            )

            # Fill in pixels in the output image iff:
            #  - The pixel in the output image is empty (alpha == 0)
            #  - The pixel in the warped ground image is not empty (alpha != 0)
            output_img = np.where(
                ((output_img[:, :, 3] == 0) & (warped_ground_img[:, :, 3] != 0))[
                    :, :, np.newaxis
                ],
                warped_ground_img,
                output_img,
            )

        return output_img

    def save(self, filename: str) -> None:
        cv2.imwrite(filename, self.get_view())
