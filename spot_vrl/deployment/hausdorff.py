# Calculate the hausdorff distances between a human demonstration trajectory and
# trial trajectories.

import argparse
import os
from typing import List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import rosbag
from genpy.message import Message
from amrl_msgs.msg import Localization2DMsg
from geometry_msgs.msg import PoseWithCovarianceStamped
from loguru import logger
import pylab
import numpy as np
import numpy.typing as npt
import scipy.spatial.distance

import matplotlib


def read_bag(bagfile: Path) -> npt.NDArray[np.float32]:
    """return value is a matrix of xy row vectors"""

    pose_topic = "/husky/inekf_estimation/pose"
    bag = rosbag.Bag(bagfile)

    n_points = bag.get_message_count(pose_topic)
    coords = np.empty((n_points, 2), np.float32)

    msg: Message
    for i, (_, msg, _) in enumerate(rosbag.Bag(bagfile).read_messages(pose_topic)):
        if pose_topic == "/localization":
            loc_pose: Localization2DMsg = msg
            coords[i, 0] = loc_pose.pose.x
            coords[i, 1] = loc_pose.pose.y
        elif pose_topic == "/husky/inekf_estimation/pose":
            ekf_pose: PoseWithCovarianceStamped = msg
            coords[i, 0] = ekf_pose.pose.pose.position.x
            coords[i, 1] = ekf_pose.pose.pose.position.y
        else:
            logger.error(f"Unrecognized topic: {pose_topic}")
            return np.zeros((1, 2), np.float32)

    return coords


def center_and_rotate(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """input: Nx2 row vector coordinates

    return Nx2 row vector coordinates

    the input may be modified in place; the return value might not be the same
    array as the input array

    translate the trajectory such that the starting point is (0, 0)

    rotate the trajectory such that the end point is (X, 0), i.e. the angular
    displacement is 0
    """
    points -= points[0]

    angle = np.arctan2(points[-1, 1] - points[0, 1], points[-1, 0] - points[0, 0])
    rotmat = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ]
    )

    # row vectors!
    points = np.transpose(rotmat @ np.transpose(points))

    if not np.allclose(points[0], 0, rtol=0, atol=1e-5):
        logger.error(f"Nonzero starting point {points[0]}")

    angle = np.arctan2(points[-1, 1] - points[0, 1], points[-1, 0] - points[0, 0])
    if not np.allclose(angle, 0, rtol=0, atol=1e-2):
        logger.error(f"Nonzero angular displacement {angle}")

    return points


def distance(bagfile: Path, human_bagfile: Path) -> None:
    traj = read_bag(bagfile)
    human_traj = read_bag(human_bagfile)

    traj = center_and_rotate(traj)
    human_traj = center_and_rotate(human_traj)

    d, index_1, index_2 = scipy.spatial.distance.directed_hausdorff(human_traj, traj)
    d2, _, _ = scipy.spatial.distance.directed_hausdorff(human_traj, traj)

    d = max(d, d2)

    logger.info(f"{bagfile}: {d:.6f}")

    ax: plt.Axes = plt.gca()

    ax.set_title(f"{bagfile}")
    ax.plot(traj[:, 0], traj[:, 1], label="model")
    ax.plot(human_traj[:, 0], human_traj[:, 1], label="human")
    ax.set_aspect("equal")
    ax.legend()
    savepath = Path(f"images/hausdorff-debug/{bagfile.stem}.png")
    os.makedirs(savepath.parent, exist_ok=True)
    plt.savefig(savepath)
    plt.close("all")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "experiment_parent_dir",
        type=Path,
        help="Path to a direct parent directory containing experimental runs.\n"
        " This directory should contain\nsubdirectories named according to the terrain model,"
        " e.g.: '6t/*.bag', '3t/*.bag', 'ganav/*.bag', 'geometric/*.bag', 'human/*.bag'",
    )
    argparser.add_argument(
        "human",
        type=Path,
        help="Path to human demonstration bagfile",
    )
    argparser.add_argument(
        "--topic", type=str, default="/localization", help="Pose topic"
    )
    args = argparser.parse_args()

    experiment_parent_dir: Path = args.experiment_parent_dir
    human_bagfile: Path = args.human
    # topic: str = args.topic

    bagfiles = []
    for root, _, files in os.walk(experiment_parent_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix == ".bag":
                bagfiles.append(path)

    bagfiles.sort()
    for bagfile in bagfiles:
        distance(bagfile, human_bagfile)

    # human_bagfile = Path(
    #     "/home/elvin/rosbags/spot/2022-11-13/gdc/human/2022-11-13-11-16-08.bag"
    # )
    # other_bagfiles = [
    #     Path("/home/elvin/rosbags/spot/2022-11-13/gdc/human/2022-11-13-11-16-08.bag"),
    #     Path("/home/elvin/rosbags/spot/2022-11-13/gdc/6t/2022-11-13-11-23-00.bag"),
    #     Path("/home/elvin/rosbags/spot/2022-11-13/gdc/3t/2022-11-13-11-45-40.bag"),
    #     Path("/home/elvin/rosbags/spot/2022-11-13/gdc/ganav/2022-11-13-12-11-45.bag"),
    #     Path(
    #         "/home/elvin/rosbags/spot/2022-11-13/gdc/geometric/2022-11-13-11-12-29.bag"
    #     ),
    # ]

    # for bagfile in other_bagfiles:
    #     distance(bagfile, human_bagfile)
