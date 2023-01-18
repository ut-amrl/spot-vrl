"""
This module provides helper functions to convert ROS messages to numpy
structures.
"""

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


import geometry_msgs.msg
import nav_msgs.msg
import sensor_msgs.msg
import spot_msgs.msg
import tf2_msgs.msg


def transform_to_affine(tform: geometry_msgs.msg.Transform) -> npt.NDArray[np.float64]:
    q = tform.rotation
    t = tform.translation

    affine = np.identity(4)
    affine[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    affine[:3, 3] = [t.x, t.y, t.z]
    return affine


def body_tform_frames(
    tf_tree: tf2_msgs.msg.TFMessage,
) -> Dict[str, npt.NDArray[np.float64]]:
    parent_lookup: Dict[str, str] = {}
    parent_tform_frame: Dict[Tuple[str, str], npt.NDArray[np.float64]] = {}

    transform_stamped: geometry_msgs.msg.TransformStamped
    for transform_stamped in tf_tree.transforms:
        parent = transform_stamped.header.frame_id
        child = transform_stamped.child_frame_id
        affine = transform_to_affine(transform_stamped.transform)

        if child == "body":
            parent, child = child, parent
            affine = np.linalg.inv(affine)

        parent_lookup[child] = parent
        parent_tform_frame[(parent, child)] = affine

    assert "body" not in parent_lookup

    # Assume the tree is fully formed at this point
    for (parent, child) in list(parent_tform_frame.keys()):
        while parent in parent_lookup:
            grandparent = parent_lookup[parent]

            affine = parent_tform_frame[(grandparent, parent)] @ parent_tform_frame.pop(
                (parent, child)
            )

            parent_tform_frame[(grandparent, child)] = affine
            parent_lookup[child] = grandparent

            parent = grandparent

    parent_tform_frame[("body", "body")] = np.identity(4)

    body_tform_frames: Dict[str, npt.NDArray[np.float64]] = {}
    for (parent, child), affine in parent_tform_frame.items():
        assert parent == "body"
        body_tform_frames[child] = affine

    return body_tform_frames


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


class TimeSyncedMessages:
    topic_types: Dict[str, Type[Any]] = {
        "/joint_states": sensor_msgs.msg.JointState,
        "/odom": nav_msgs.msg.Odometry,
        "/spot/odometry/twist": geometry_msgs.msg.TwistWithCovarianceStamped,
        "/spot/status/battery_states": spot_msgs.msg.BatteryStateArray,
        "/spot/status/feet": spot_msgs.msg.FootStateArray,
        "/tf": tf2_msgs.msg.TFMessage,
    }

    def __init__(self) -> None:
        # These MUST be the named using suffix/basename of the corresponding topic names
        self.joint_states: sensor_msgs.msg.JointState = None
        self.odom: nav_msgs.msg.Odometry = None
        self.twist: geometry_msgs.msg.TwistWithCovarianceStamped = None
        self.battery_states: spot_msgs.msg.BatteryStateArray = None
        self.feet: spot_msgs.msg.FootStateArray = None
        self.tf: tf2_msgs.msg.TFMessage = None

    def set(self, topic: str, msg: Any) -> None:
        if topic not in self.topic_types:
            return

        # It seems like temporary types are being used, so this
        # check is ineffective. Assume the caller provided the types
        # correctly

        # if type(msg) != self.topic_types[topic]:
        #     logger.error(
        #         f"Expected type ({self.topic_types[topic]}) for topic {topic},"
        #         f" got {type(msg)}"
        #     )
        #     return

        field_name = topic.split("/")[-1]
        setattr(self, field_name, msg)

    def valid(self) -> bool:
        return (
            self.joint_states is not None
            and self.odom is not None
            and self.twist is not None
            and self.battery_states is not None
            and self.feet is not None
            and self.tf is not None
        )
