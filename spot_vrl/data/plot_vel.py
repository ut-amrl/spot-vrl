"""Plots line graphs of the linear and angular velocities of a dataset.

Primarily designed as a sanity check for data reading and plotting.

Invocation from the project's root directory:
    python3 -m spot_vrl.data.plot_vel [...]
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.geometry_pb2 import SE3Velocity, Vec3
from bosdyn.api.robot_state_pb2 import RobotState, RobotStateResponse, KinematicState
from bosdyn.bddf import DataReader, ProtobufReader


class VelocityData:
    """Helper class for storing various velocities."""

    def __init__(self, ts_sec: List[float], v: List[SE3Velocity]) -> None:
        assert len(ts_sec) == len(v)
        N = len(v)

        self.timestamps = np.array(ts_sec)
        self.timestamps -= self.timestamps.min()

        # first pass to filter out points and shrink N
        for i in range(N):
            v_lin: Vec3 = v[i].linear
            ground = np.linalg.norm((v_lin.x, v_lin.y))
            if self.timestamps[i] > 10 and ground < 0.15:
                N = i
                self.timestamps = self.timestamps[:N]
                break

        self.ground = np.empty(N)
        self.depth = np.empty(N)
        self.roll = np.empty(N)
        self.pitch = np.empty(N)
        self.yaw = np.empty(N)

        for i in range(N):
            v_lin: Vec3 = v[i].linear
            v_ang: Vec3 = v[i].angular

            self.ground[i] = np.linalg.norm((v_lin.x, v_lin.y))
            self.depth[i] = v_lin.z
            self.roll[i] = v_ang.x
            self.pitch[i] = v_ang.y
            self.yaw[i] = v_ang.z


def read_velocities(filename: str) -> VelocityData:
    """Returns VelocityData read from the input `filename`."""
    timestamps: list[float] = []
    velocities: List[SE3Velocity] = []

    data_reader = DataReader(None, filename)
    proto_reader = ProtobufReader(data_reader)

    series_idx: int = proto_reader.series_index("bosdyn.api.RobotStateResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_idx)
    num_msgs = len(series_block_index.block_entries)

    for msg_idx in range(num_msgs):
        timestamp_nsec: int
        response: RobotStateResponse
        _, timestamp_nsec, response = proto_reader.get_message(
            series_idx, RobotStateResponse, msg_idx
        )

        robot_state: RobotState = response.robot_state
        kin_state: KinematicState = robot_state.kinematic_state
        v: SE3Velocity = kin_state.velocity_of_body_in_odom

        velocities.append(v)
        timestamps.append(float(timestamp_nsec) * 1e-9)

    return VelocityData(timestamps, velocities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open")
    parser.add_argument("--gui", action="store_true", help="Open the matplotlib GUI")

    options = parser.parse_args()
    data = read_velocities(options.filename)

    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = GridSpec(2, 6, figure=fig)

    ax1: plt.Axes = fig.add_subplot(gs[0, :3])  # ground speed
    ax2: plt.Axes = fig.add_subplot(gs[0, 3:])  # depth velocity
    ax3: plt.Axes = fig.add_subplot(gs[1, :2])  # roll velocity
    ax4: plt.Axes = fig.add_subplot(gs[1, 2:4])  # pitch velocity
    ax5: plt.Axes = fig.add_subplot(gs[1, 4:])  # yaw velocity

    # add basic plot data, labels
    fig.suptitle(f"Trajectory Velocities ({options.filename})")

    ax1.plot(data.timestamps, data.ground, color="k")
    ax1.set_ylim(top=1.4)
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Ground Speed", fontsize="medium")

    ax2.plot(data.timestamps, data.depth, color="tab:brown")
    ax2.set_ylim(bottom=-0.25, top=0.25)
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Z-Axis Velocity", fontsize="medium")

    ax3.plot(data.timestamps, data.roll, color="tab:red")
    ax3.set_ylim(bottom=-1, top=1)
    ax3.set_ylabel("Angular Velocity (rad/s)", fontsize="large")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Roll", fontsize="medium")

    ax4.plot(data.timestamps, data.pitch, color="tab:green")
    ax4.set_ylim(bottom=-1, top=1)
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Pitch", fontsize="medium")

    ax5.plot(data.timestamps, data.yaw, color="tab:blue")
    ax5.set_ylim(bottom=-0.25, top=0.25)
    ax5.set_xlabel("Time (s)")
    ax5.set_title("Yaw", fontsize="medium")

    # fill regions of interest where the robot is not turning around
    fill_regions = abs(data.yaw) < 0.4

    def fill_roi(ax: plt.Axes) -> None:
        ybot, ytop = ax.get_ylim()
        ax.fill_between(
            data.timestamps,
            ybot,
            ytop,
            where=fill_regions,
            color="tab:gray",
            alpha=0.25,
        )

    # fill_roi(ax1)
    # fill_roi(ax2)
    # fill_roi(ax3)
    # fill_roi(ax4)
    # fill_roi(ax5)

    out_stem = f"{Path(__file__).stem}-{Path(options.filename).stem}.pdf"
    out_path = Path(options.filename).parent / out_stem

    plt.savefig(out_path, format="pdf", dpi=120)
    print(f"Plot saved to: {out_path}")

    if options.gui:
        plt.show()


if __name__ == "__main__":
    main()
