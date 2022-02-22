from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from bosdyn.api import geometry_pb2
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.robot_state_pb2 import FootState, RobotStateResponse
from bosdyn.bddf import DataReader, ProtobufReader
from loguru import logger


class ImuData:
    """Storage container for non-visual Spot sensor data of interest.

    This class stores a time-ordered sequence of deconstructed RobotState
    messages from a BDDF file. Only a subset of RobotState is stored; these
    fields can be found in the constructor for the Datum inner class.

    The full sequence of each field can be accessed as numpy arrays using the
    property fields of the ImuData class. Each item in the sequence is
    represented as a column vector.
    """

    class Datum:
        joint_order: Dict[str, int] = {
            "fl.hx": 0,
            "fl.hy": 1,
            "fl.kn": 2,
            "fr.hx": 3,
            "fr.hy": 4,
            "fr.kn": 5,
            "hl.hx": 6,
            "hl.hy": 7,
            "hl.kn": 8,
            "hr.hx": 9,
            "hr.hy": 10,
            "hr.kn": 11,
        }

        def __init__(self, response: RobotStateResponse) -> None:
            self.ts: np.float64 = np.float64(0)
            """Robot system timestamp of the data in seconds."""

            self.power: np.float32 = np.float32(0)
            """Power consumption measured by battery discharge."""

            self.joint_pos: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
            """Angular position of each leg joint."""

            self.joint_vel: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
            """Velocity (angular?) of each leg joint."""

            self.joint_acc: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
            """Acceleration (angular?) of each leg joint."""

            self.linear_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
            """Linear velocity measured by odometry."""

            self.angular_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
            """Angular velocity measured by odometry."""

            self.foot_slip_dist: np.float32 = np.float32(0)
            """Mean foot slip distance (vector norm)."""

            self.foot_slip_vel: np.float32 = np.float32(0)
            """Mean foot slip velocity (vector norm)."""

            self.foot_depth_mean: np.float32 = np.float32(0)
            """Mean of foot depth values in the ground plane estimate."""

            self.foot_depth_std: np.float32 = np.float32(0)
            """Mean of foot depth uncertainties in the ground plane estimate. """

            robot_state = response.robot_state
            kinematic_state = robot_state.kinematic_state

            # TODO(eyang): should this be the kinematic state acquisition
            # timestamp rather than the response timestamp? The kinematic state
            # seems to be acquired at about 12 Hz, so RobotStateResponses
            # acquired at a higher frequency may contain the same data.
            self.ts = (
                response.header.response_timestamp.seconds
                + response.header.response_timestamp.nanos * 1e-9
            )
            self.power = np.float32(0)
            for battery in robot_state.battery_states:
                if battery.current.value >= 0:
                    logger.warning(
                        f"Expected negative current reading, got {battery.current.value}"
                    )
                self.power += -battery.current.value * battery.voltage.value

            if len(kinematic_state.joint_states) != 12:
                logger.warning(
                    f"Expected 12 joint states, got {len(kinematic_state.joint_states)}"
                )
            for joint in kinematic_state.joint_states:
                joint_idx = self.joint_order[joint.name]
                self.joint_pos[joint_idx] = joint.position.value
                self.joint_vel[joint_idx] = joint.velocity.value
                self.joint_acc[joint_idx] = joint.acceleration.value

            odom_vel = kinematic_state.velocity_of_body_in_odom
            self.linear_vel[:] = odom_vel.linear.x, odom_vel.linear.y, odom_vel.linear.z
            self.angular_vel[:] = (
                odom_vel.angular.x,
                odom_vel.angular.y,
                odom_vel.angular.z,
            )

            feet_in_contact = 0
            for foot in robot_state.foot_state:
                if foot.contact != FootState.CONTACT_MADE:
                    continue
                feet_in_contact += 1
                terrain = foot.terrain
                if terrain.frame_name != "odom":
                    logger.warning(
                        f"Expected 'odom' as foot reference frame, got {terrain.frame_name}"
                    )

                def _vec3_norm(v: geometry_pb2.Vec3) -> np.float32:
                    return np.linalg.norm((v.x, v.y, v.z)).astype(np.float32)

                self.foot_slip_dist += _vec3_norm(terrain.foot_slip_distance_rt_frame)
                self.foot_slip_vel += _vec3_norm(terrain.foot_slip_velocity_rt_frame)
                self.foot_depth_mean += terrain.visual_surface_ground_penetration_mean
                self.foot_depth_std += terrain.visual_surface_ground_penetration_std

            if feet_in_contact != 0:
                self.foot_slip_dist /= feet_in_contact
                self.foot_slip_vel /= feet_in_contact
                self.foot_depth_mean /= feet_in_contact
                self.foot_depth_std /= feet_in_contact
            else:
                logger.warning("Spot is flying (no feet made contact with the ground)")

    def __init__(self, filename: Union[str, Path]) -> None:
        """
        Args:
            filename (str | Path): Path to a BDDF file.
        """
        self._data: List["ImuData.Datum"] = []

        data_reader = DataReader(None, str(filename))
        proto_reader = ProtobufReader(data_reader)

        series_index: int = proto_reader.series_index("bosdyn.api.RobotStateResponse")
        series_block_index: SeriesBlockIndex = data_reader.series_block_index(
            series_index
        )
        num_msgs = len(series_block_index.block_entries)

        for msg_idx in range(num_msgs):
            _, _, response = proto_reader.get_message(
                series_index, RobotStateResponse, msg_idx
            )

            self._data.append(ImuData.Datum(response))

        if not all(
            self._data[i].ts < self._data[i + 1].ts for i in range(len(self._data) - 1)
        ):
            logger.warning("Data sequence is not sorted by ascending ts.")

    @property
    def timestamp_sec(self) -> npt.NDArray[np.float64]:
        return np.array([d.ts for d in self._data], dtype=np.float64)

    @property
    def power(self) -> npt.NDArray[np.float32]:
        return np.array([d.power for d in self._data], dtype=np.float32)

    @property
    def joint_pos(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((12, len(self._data)), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[:, i] = d.joint_pos
        return tensor

    @property
    def joint_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((12, len(self._data)), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[:, i] = d.joint_vel
        return tensor

    @property
    def joint_acc(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((12, len(self._data)), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[:, i] = d.joint_acc
        return tensor

    @property
    def linear_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((3, len(self._data)), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[:, i] = d.linear_vel
        return tensor

    @property
    def angular_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((3, len(self._data)), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[:, i] = d.angular_vel
        return tensor

    @property
    def foot_slip_dist(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_slip_dist for d in self._data], dtype=np.float32)

    @property
    def foot_slip_vel(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_slip_vel for d in self._data], dtype=np.float32)

    @property
    def foot_depth_mean(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_depth_mean for d in self._data], dtype=np.float32)

    @property
    def foot_depth_std(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_depth_std for d in self._data], dtype=np.float32)

    @property
    def all_data(self) -> npt.NDArray[np.float32]:
        """All sensor data, excluding timestamps."""
        return np.vstack(
            (
                self.power,
                self.joint_pos,
                self.joint_vel,
                self.joint_acc,
                self.linear_vel,
                self.angular_vel,
                self.foot_slip_dist,
                self.foot_slip_vel,
                self.foot_depth_mean,
                self.foot_depth_std,
            )
        )
