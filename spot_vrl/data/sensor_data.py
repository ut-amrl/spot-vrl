from collections import defaultdict
import sys
from functools import lru_cache, cached_property
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from bosdyn.api import geometry_pb2
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.api.robot_state_pb2 import FootState, RobotStateResponse
from bosdyn.bddf import DataReader, ProtobufReader
from loguru import logger
from spot_vrl.data import proto_to_numpy

if sys.platform == "linux":
    import geometry_msgs.msg
    import spot_msgs.msg
    import rosbag
    from spot_vrl.data import ros_to_numpy


class SensorMetrics:
    joint_order: ClassVar[Dict[str, int]] = {
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

    # The spot_ros library uses "friendly" joint names
    ros_joint_order: ClassVar[Dict[str, int]] = {
        "front_left_hip_x": 0,
        "front_left_hip_y": 1,
        "front_left_knee": 2,
        "front_right_hip_x": 3,
        "front_right_hip_y": 4,
        "front_right_knee": 5,
        "rear_left_hip_x": 6,
        "rear_left_hip_y": 7,
        "rear_left_knee": 8,
        "rear_right_hip_x": 9,
        "rear_right_hip_y": 10,
        "rear_right_knee": 11,
    }

    def __init__(self) -> None:
        self.ts: np.float64 = np.float64(0)
        """Robot system timestamp of the data in seconds."""

        self.tf_tree: FrameTreeSnapshot = FrameTreeSnapshot()
        """The transform tree from this state."""

        self.body_tform_frames: Dict[str, npt.NDArray[np.float64]] = {}
        """The parsed transform tree from this state. Stores the transformation
        from the body frame to all other frames."""

        self.power: np.float32 = np.float32(0)
        """Power consumption measured by battery discharge."""

        self.joint_pos: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
        """Angular position of each leg joint."""

        self.joint_vel: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
        """Velocity (angular?) of each leg joint."""

        self.joint_load: npt.NDArray[np.float32] = np.zeros(12, dtype=np.float32)
        """Load (torque, Newton-meters) of each leg joint."""

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
        """Mean of foot depth uncertainties in the ground plane estimate."""

    @classmethod
    def from_bosdyn(cls, response: RobotStateResponse) -> "SensorMetrics":
        metrics = cls()

        robot_state = response.robot_state
        kinematic_state = robot_state.kinematic_state

        metrics.ts = (
            kinematic_state.acquisition_timestamp.seconds
            + kinematic_state.acquisition_timestamp.nanos * 1e-9
        )

        metrics.tf_tree = kinematic_state.transforms_snapshot
        for frame in metrics.tf_tree.child_to_parent_edge_map.keys():
            metrics.body_tform_frames[frame] = proto_to_numpy.body_tform_frame(
                metrics.tf_tree, frame
            )

        metrics.power = np.float32(0)
        for battery in robot_state.battery_states:
            if battery.current.value >= 0:
                logger.warning(
                    f"Expected negative current reading, got {battery.current.value}"
                )
            metrics.power += -battery.current.value * battery.voltage.value

        if len(kinematic_state.joint_states) != 12:
            logger.warning(
                f"Expected 12 joint states, got {len(kinematic_state.joint_states)}"
            )
        for joint in kinematic_state.joint_states:
            joint_idx = metrics.joint_order[joint.name]
            metrics.joint_pos[joint_idx] = joint.position.value
            metrics.joint_vel[joint_idx] = joint.velocity.value
            metrics.joint_load[joint_idx] = joint.load.value

        odom_vel = kinematic_state.velocity_of_body_in_odom
        metrics.linear_vel[:] = (
            odom_vel.linear.x,
            odom_vel.linear.y,
            odom_vel.linear.z,
        )
        metrics.angular_vel[:] = (
            odom_vel.angular.x,
            odom_vel.angular.y,
            odom_vel.angular.z,
        )

        # Rotate the odometry velocities into the robot frame
        body_tform_odom = proto_to_numpy.body_tform_frame(metrics.tf_tree, "odom")
        metrics.linear_vel = body_tform_odom[:3, :3] @ metrics.linear_vel
        metrics.angular_vel = body_tform_odom[:3, :3] @ metrics.angular_vel

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

            metrics.foot_slip_dist += _vec3_norm(terrain.foot_slip_distance_rt_frame)
            metrics.foot_slip_vel += _vec3_norm(terrain.foot_slip_velocity_rt_frame)
            metrics.foot_depth_mean += terrain.visual_surface_ground_penetration_mean
            metrics.foot_depth_std += terrain.visual_surface_ground_penetration_std

        if feet_in_contact != 0:
            metrics.foot_slip_dist /= feet_in_contact
            metrics.foot_slip_vel /= feet_in_contact
            metrics.foot_depth_mean /= feet_in_contact
            metrics.foot_depth_std /= feet_in_contact
        else:
            logger.warning("Spot is flying (no feet made contact with the ground)")

        return metrics

    @classmethod
    def from_ros(cls, msgs: ros_to_numpy.TimeSyncedMessages) -> "SensorMetrics":
        metrics = cls()

        metrics.ts = msgs.odom.header.stamp.to_sec()
        metrics.body_tform_frames = ros_to_numpy.body_tform_frames(msgs.tf)

        battery: spot_msgs.msg.BatteryState
        for battery in msgs.battery_states.battery_states:
            if battery.current >= 0:
                logger.warning(
                    f"Expected negative current reading, got {battery.current}"
                )
            metrics.power += -battery.current * battery.voltage

        if len(msgs.joint_states.name) != 12:
            logger.warning(
                f"Expected 12 joint states, got {len(msgs.joint_states.name)}"
            )
        joint_name: str
        for i, joint_name in enumerate(msgs.joint_states.name):
            joint_idx = metrics.ros_joint_order[joint_name]
            metrics.joint_pos[joint_idx] = msgs.joint_states.position[i]
            metrics.joint_vel[joint_idx] = msgs.joint_states.velocity[i]
            metrics.joint_load[joint_idx] = msgs.joint_states.effort[i]

        odom_vel = msgs.twist.twist.twist
        metrics.linear_vel[:] = (
            odom_vel.linear.x,
            odom_vel.linear.y,
            odom_vel.linear.z,
        )
        metrics.angular_vel[:] = (
            odom_vel.angular.x,
            odom_vel.angular.y,
            odom_vel.angular.z,
        )

        body_tform_odom = metrics.body_tform_frames["odom"]
        metrics.linear_vel = body_tform_odom[:3, :3] @ metrics.linear_vel
        metrics.angular_vel = body_tform_odom[:3, :3] @ metrics.angular_vel

        feet_in_contact = 0
        foot_state: spot_msgs.msg.FootState
        for foot_state in msgs.feet.states:
            if foot_state.contact != spot_msgs.msg.FootState.CONTACT_MADE:
                continue
            feet_in_contact += 1
            if foot_state.frame_name != "odom":
                logger.warning(
                    f"Expected 'odom' as foot reference frame, got {foot_state.frame_name}"
                )

            def _vec3_norm(v: geometry_msgs.msg.Point) -> np.float32:
                return np.linalg.norm((v.x, v.y, v.z)).astype(np.float32)

            metrics.foot_slip_dist += _vec3_norm(foot_state.foot_slip_distance_rt_frame)
            metrics.foot_slip_vel += _vec3_norm(foot_state.foot_slip_velocity_rt_frame)
            metrics.foot_depth_mean += foot_state.visual_surface_ground_penetration_mean
            metrics.foot_depth_std += foot_state.visual_surface_ground_penetration_std

        if feet_in_contact != 0:
            metrics.foot_slip_dist /= feet_in_contact
            metrics.foot_slip_vel /= feet_in_contact
            metrics.foot_depth_mean /= feet_in_contact
            metrics.foot_depth_std /= feet_in_contact
        else:
            logger.warning("Spot is flying (no feet made contact with the ground)")

        return metrics


class ImuData:
    """Storage container for non-visual Spot sensor data of interest.

    This class stores a time-ordered sequence of deconstructed RobotState
    messages from a BDDF file. Only a subset of RobotState is stored; these
    fields can be found in the constructor for the Datum inner class.

    The full sequence of each field can be accessed as numpy arrays using the
    property fields of the ImuData class. Each item in the sequence is
    represented as a row vector.
    """

    def __init__(self, filename: Union[str, Path]) -> None:
        """
        Args:
            filename (str | Path): Path to a BDDF file.
        """
        self._data: List[SensorMetrics] = []
        self._path: Path = Path(filename)

        if self._path.suffix == ".bddf":
            self._init_from_bddf()
        elif self._path.suffix == ".bag":
            self._init_from_rosbag()
        else:
            logger.error(f"Unrecognized file format {self._path}")
            raise ValueError

        if not all(
            self._data[i].ts < self._data[i + 1].ts for i in range(len(self._data) - 1)
        ):
            logger.warning("Data sequence is not sorted by ascending ts.")

    def _init_from_bddf(self) -> None:
        data_reader = DataReader(None, str(self._path))
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

            self._data.append(SensorMetrics.from_bosdyn(response))

    def _init_from_rosbag(self) -> None:
        # times don't match so the best we can do is use indices as keys
        synced_msgs: Dict[int, ros_to_numpy.TimeSyncedMessages] = defaultdict(
            ros_to_numpy.TimeSyncedMessages  # type: ignore
        )
        topic_count: Dict[str, int] = defaultdict(int)

        bag = rosbag.Bag(str(self._path))
        for topic, msg, _ in bag.read_messages():
            key = topic_count[topic]
            synced_msgs[key].set(topic, msg)
            topic_count[topic] += 1

        for v in synced_msgs.values():
            if v.valid():
                self._data.append(SensorMetrics.from_ros(v))

    def __len__(self) -> int:
        return len(self._data)

    @cached_property
    def timestamp_sec(self) -> npt.NDArray[np.float64]:
        return np.array([d.ts for d in self._data], dtype=np.float64)

    @cached_property
    def power(self) -> npt.NDArray[np.float32]:
        return np.array([d.power for d in self._data], dtype=np.float32)

    @cached_property
    def joint_pos(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((len(self._data), 12), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[i] = d.joint_pos
        return tensor

    @cached_property
    def joint_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((len(self._data), 12), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[i] = d.joint_vel
        return tensor

    @cached_property
    def joint_load(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((len(self._data), 12), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[i] = d.joint_load
        return tensor

    @cached_property
    def linear_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((len(self._data), 3), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[i] = d.linear_vel
        return tensor

    @cached_property
    def angular_vel(self) -> npt.NDArray[np.float32]:
        tensor = np.empty((len(self._data), 3), dtype=np.float32)
        for i, d in enumerate(self._data):
            tensor[i] = d.angular_vel
        return tensor

    @cached_property
    def foot_slip_dist(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_slip_dist for d in self._data], dtype=np.float32)

    @cached_property
    def foot_slip_vel(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_slip_vel for d in self._data], dtype=np.float32)

    @cached_property
    def foot_depth_mean(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_depth_mean for d in self._data], dtype=np.float32)

    @cached_property
    def foot_depth_std(self) -> npt.NDArray[np.float32]:
        return np.array([d.foot_depth_std for d in self._data], dtype=np.float32)

    @cached_property
    def all_sensor_data(self) -> npt.NDArray[np.float32]:
        return np.hstack(
            (
                self.power[:, np.newaxis],
                self.joint_pos,
                self.joint_vel,
                self.joint_load,
                self.linear_vel,
                self.angular_vel,
                self.foot_slip_dist[:, np.newaxis],
                self.foot_slip_vel[:, np.newaxis],
                self.foot_depth_mean[:, np.newaxis],
                self.foot_depth_std[:, np.newaxis],
            )
        )

    @lru_cache(maxsize=None)
    def tforms(
        self,
        reference_frame: str,
        frame_of_interest: str,
    ) -> npt.NDArray[np.float32]:
        """Computes the 3D affine transforms between two frames.

        Specifically, this function computes the transforms A_tform_B, where
            A = reference_frame
            B = frame_of_interest

            A_tform_B @ (point in frame B) => (point in frame A)
            inv(A_tform_B) @ (point in frame A) => (point in frame B)
            A_tform_B @ B_tform_C => A_tform_C

        Args:
            reference_frame (str): The name of the reference frame.
            frame_of_interest (str): The name of the frame of interest.

            RobotStateResponses do not contain the full transform tree.
            In general, these frame names are present:
                "odom"
                "gpe"
                "body"

        Returns:
            npt.NDArray[np.float32]:
                An (N, 4, 4) array of 3D affine transformation matrices,
                where N is the number of state observations.
        """
        transforms: npt.NDArray[np.float32] = np.empty(
            (len(self), 4, 4), dtype=np.float32
        )

        for i, datum in enumerate(self._data):
            # "body" is the root of the transform tree
            # body_tform_ref = proto_to_numpy.body_tform_frame(
            #     datum.tf_tree, reference_frame
            # )
            # body_tform_interest = proto_to_numpy.body_tform_frame(
            #     datum.tf_tree, frame_of_interest
            # )
            body_tform_ref = datum.body_tform_frames[reference_frame]
            body_tform_interest = datum.body_tform_frames[frame_of_interest]
            ref_tform_interest = np.linalg.inv(body_tform_ref) @ body_tform_interest
            transforms[i] = ref_tform_interest.astype(np.float32)

        return transforms

    def query_time_range(
        self,
        prop: npt.NDArray[np.float32],
        start: Union[float, np.float_] = 0,
        end: Union[float, np.float_] = np.inf,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float32]]:
        """Queries a time range of the specified property.

        Args:
            prop (npt.NDArray[np.float32]): A numpy array returned by one of
                this class' properties.
            start (float): The start of the time range (seconds, inclusive).
            end (float): The end of the time range (seconds, exclusive).

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float32]]:
                A tuple containing the timestamps and data corresponding to
                the time range.

        Raises:
            ValueError: Invalid time range or property matrix.

        Examples:
            >>> imu = ImuData(...)
            >>> timestamps, data = imu.query_time_range(imu.all_sensor_data, 1000, 2000)
            >>> timestamps, power = imu.query_time_range(imu.power, 1000, 2000)
        """
        if start >= end:
            raise ValueError("start >= end")
        if prop.shape[0] != len(self._data):
            raise ValueError("Number of rows does not match internal list.")

        if start > self.timestamp_sec[-1] + 1 or end < self.timestamp_sec[0] - 1:
            logger.warning(
                f"Specified time range ({start}, {end}) and dataset ({self._path}) are disjoint "
            )

        start_i = int(np.searchsorted(self.timestamp_sec, start))
        end_i = int(np.searchsorted(self.timestamp_sec, end))
        ts_range = self.timestamp_sec[start_i:end_i]
        prop_range = prop[start_i:end_i]

        if ts_range.size == 0:
            logger.warning("Returning empty arrays")
        return ts_range, prop_range
