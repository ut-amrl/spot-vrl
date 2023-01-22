from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import numpy.typing as npt
import tqdm

from spot_vrl.data.sensor_data import SpotSensorData
from spot_vrl.data.image_data import BEVImageSequence, Image


@dataclass
class Datum:
    image: npt.NDArray[np.uint8]
    """An RGB BEV image."""

    odom: npt.NDArray[np.float32]
    """
    Full 4x4 Affine 3D matrix describing the location of the robot in the odom
    frame.
    """

    spot_history: npt.NDArray[np.float32]
    """
    A short history of Spot proprioceptive data before the image was captured.

    The 0-axis represents time. The 1-axis represents different sensor metrics.
    (see SpotSensorData.all_sensor_data)
    """


class SynchronizedData:
    """
    Storage container with approximately synchronized Spot, Odometry, and Image
    data.
    """

    def __init__(
        self, filename: Union[str, Path], sensor_history_sec: float = 1.0
    ) -> None:
        self.data: List[Datum] = []
        """
        List of points along the trajectory ordered by timestamp. Each point
        along the trajectory corresponds to an image capture.
        """

        spot_container = SpotSensorData(filename)
        image_container = BEVImageSequence(filename)

        image_timestamp: np.float64
        image: Image
        for image_timestamp, image in tqdm.tqdm(
            image_container, desc="Loading SyncedData", total=len(image_container)
        ):
            # Skip this image if there does not exist a large enough data window
            if image_timestamp - sensor_history_sec < spot_container.timestamp_sec[0]:
                continue

            if image_timestamp > spot_container.timestamp_sec[-1]:
                break

            # Query the data window immediately before this image was taken.
            _, spot_history = spot_container.query_time_range(
                spot_container.all_sensor_data,
                start=image_timestamp - sensor_history_sec,
                end=image_timestamp,
            )

            # Query the first odometry pose after this image was taken
            _, odom_poses = spot_container.query_time_range(
                spot_container.tforms("odom", "body"), start=image_timestamp
            )
            this_odom_pose = odom_poses[0]

            self.data.append(Datum(image.decoded_image(), this_odom_pose, spot_history))
