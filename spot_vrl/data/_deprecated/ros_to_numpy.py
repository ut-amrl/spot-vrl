"""
This module provides helper functions to convert ROS messages to numpy
structures.
"""

import warnings
from typing import List, Union

import numpy as np
import numpy.typing as npt

import sensor_msgs.msg


def est_kinect_rot(
    imu: Union[sensor_msgs.msg.Imu, List[sensor_msgs.msg.Imu]]
) -> npt.NDArray[np.float64]:
    """Estimates the rotation of an Azure Kinect DK using IMU readings taken
    at rest.

    Uses Rodrigues' rotation formula to calculate the rotation from the reported
    gravity vector to the unit vector pointing straight up from the body of the
    Kinect.

    IMU readings from the Kinect use the Z-towards-ground coordinate frame.
    (Thus, when a Kinect is flat and at rest, acceleration due to gravity is
    reported as approximately [0, 0, -9.8])
    """
    warnings.warn(
        "Homography transformation code has been ported to C++ to run during"
        " data collection (https://github.com/ut-amrl/local_rgb_map)."
        " This function is no longer maintained.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    readings: List[sensor_msgs.msg.Imu]
    if type(imu) is list:
        readings = imu
    else:
        readings = [imu]

    accels = []
    for reading in readings:
        accels.append(
            [
                reading.linear_acceleration.x,
                reading.linear_acceleration.y,
                reading.linear_acceleration.z,
            ]
        )
    v_g = np.median(np.array(accels, dtype=np.float64), axis=0)

    unit_up = np.array([0, 0, -1], dtype=np.float64)

    v_g /= np.linalg.norm(v_g)
    cross = np.cross(v_g, unit_up)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(v_g, unit_up)
    cross /= sin_a
    K = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ]
    )
    return np.identity(3) + sin_a * K + (1 - cos_a) * K @ K  # type: ignore
