"""Plots violin plots of various sensor data in four datasets.


"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.geometry_pb2 import SE3Velocity, Vec3
from bosdyn.api.robot_state_pb2 import (
    RobotState,
    RobotStateResponse,
    KinematicState,
    FootState,
)
from bosdyn.bddf import DataReader, ProtobufReader


class SensorData:
    """Helper class for storing various sensor data types."""

    def __init__(self, filename: str) -> None:
        timestamps, robot_states = self._get_robot_states(filename)
        assert len(timestamps) == len(robot_states)

        self.path = Path(filename)
        self.timestamps = np.array(timestamps)
        self.timestamps -= min(self.timestamps)
        self.power: npt.NDArray[np.float64]
        self.ground_mu: npt.NDArray[np.float64]
        self.slip_dist: npt.NDArray[np.float64]
        self.slip_vel: npt.NDArray[np.float64]
        self.depth_mean: npt.NDArray[np.float64]
        self.depth_std: npt.NDArray[np.float64]

        power: list[float] = []
        ground_mu: list[float] = []
        slip_dist: list[np.float64] = []
        slip_vel: list[np.float64] = []
        depth_mean: list[float] = []
        depth_std: list[float] = []

        for state in robot_states:
            assert len(state.battery_states) == 1
            battery_state = state.battery_states[0]
            # current is negative if the battery is discharging
            power.append(-battery_state.current.value * battery_state.voltage.value)

            for foot_state in state.foot_state:
                if foot_state.contact != FootState.Contact.CONTACT_MADE:
                    terrain = foot_state.terrain
                    ground_mu.append(terrain.ground_mu_est)
                    assert terrain.frame_name == "odom"
                    dist = terrain.foot_slip_distance_rt_frame
                    slip_dist.append(np.linalg.norm((dist.x, dist.y, dist.z)))
                    vel = terrain.foot_slip_velocity_rt_frame
                    slip_vel.append(np.linalg.norm((vel.x, vel.y, vel.z)))
                    depth_mean.append(terrain.visual_surface_ground_penetration_mean)
                    depth_std.append(terrain.visual_surface_ground_penetration_std)

        self.power = np.array(power)
        self.ground_mu = np.array(ground_mu)
        self.slip_dist = np.array(slip_dist)
        self.slip_vel = np.array(slip_vel)
        self.depth_mean = np.array(depth_mean)
        self.depth_std = np.array(depth_std)

    def _get_robot_states(self, filename: str) -> Tuple[List[float], List[RobotState]]:
        data_reader = DataReader(None, filename)
        proto_reader = ProtobufReader(data_reader)

        series_idx: int = proto_reader.series_index("bosdyn.api.RobotStateResponse")
        series_block_index: SeriesBlockIndex = data_reader.series_block_index(
            series_idx
        )
        num_msgs = len(series_block_index.block_entries)

        timestamps: List[float] = []
        robot_states: List[RobotState] = []

        first_timestamp = 0

        for msg_idx in range(num_msgs):
            timestamp_nsec: int
            response: RobotStateResponse
            _, timestamp_nsec, response = proto_reader.get_message(
                series_idx, RobotStateResponse, msg_idx
            )

            timestamp = float(timestamp_nsec) * 1e-9
            if first_timestamp == 0:
                first_timestamp = timestamp
            timestamp -= first_timestamp

            robot_state: RobotState = response.robot_state
            kin_state: KinematicState = robot_state.kinematic_state
            v: SE3Velocity = kin_state.velocity_of_body_in_odom
            v_lin: Vec3 = v.linear
            v_ground = np.linalg.norm((v_lin.x, v_lin.y))

            # suboptimal datasets, terminate when the robot is about to turn around
            if timestamp > 10 and v_ground < 0.15:
                break

            timestamps.append(timestamp)
            robot_states.append(robot_state)

        return timestamps, robot_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open the matplotlib GUI")

    options = parser.parse_args()

    concrete = SensorData("data/concrete.bddf")
    grass = SensorData("data/grass.bddf")
    small_rocks = SensorData("data/small-rocks.bddf")
    large_rocks = SensorData("data/large-rocks.bddf")

    data = [concrete, grass, small_rocks, large_rocks]
    pos = [-1, -2, -3, -4]
    category = ["concrete", "grass", "small rocks", "large rocks"]
    colors = ["black", "tab:green", "tab:orange", "tab:brown"]

    fig: Figure
    fig, axs = plt.subplots(1, 6, sharey=True, constrained_layout=True)
    fig.set_size_inches(16, 7)

    vp0 = axs[0].violinplot([datum.power for datum in data], positions=pos, vert=False)
    axs[0].set_yticks(pos, labels=category)
    axs[0].set_xlabel("Power (W)")
    axs[0].set_title("Power Consumption")

    vp1 = axs[1].violinplot(
        [datum.ground_mu for datum in data], positions=pos, vert=False
    )
    axs[1].set_xlabel("COF")
    axs[1].set_title("Ground Coefficient of Friction")

    vp2 = axs[2].violinplot(
        [datum.slip_dist for datum in data], positions=pos, vert=False
    )
    axs[2].set_xlabel("Distance (m)")
    axs[2].set_title("Slip Distance")

    vp3 = axs[3].violinplot(
        [datum.slip_vel for datum in data], positions=pos, vert=False
    )
    axs[3].set_xlabel("Velocity (m/s)")
    axs[3].set_title("Slip Velocity")

    vp4 = axs[4].violinplot(
        [datum.depth_mean for datum in data], positions=pos, vert=False
    )
    axs[4].set_xlabel("Distance (m)")
    axs[4].set_title("Ground Penetration (mean)")

    vp5 = axs[5].violinplot(
        [datum.depth_std for datum in data], positions=pos, vert=False
    )
    axs[5].set_xlabel("Distance (m)")
    axs[5].set_title("Ground Penetration (stddev)")

    def set_violin_colors(vp: Dict[str, Any]) -> None:
        bodies = vp["bodies"]
        for i, body in enumerate(bodies):
            body.set_color(colors[i])

        line_keys = ["cmins", "cmaxes", "cbars"]
        for key in line_keys:
            lines = vp[key]
            lines.set_color(colors)

    set_violin_colors(vp0)
    set_violin_colors(vp1)
    set_violin_colors(vp2)
    set_violin_colors(vp3)
    set_violin_colors(vp4)
    set_violin_colors(vp5)

    out_path = "data/plot_sensor-all.pdf"
    plt.savefig(out_path, format="pdf")
    print(f"Plot saved to: {out_path}")

    if options.gui:
        plt.show()


if __name__ == "__main__":
    main()
