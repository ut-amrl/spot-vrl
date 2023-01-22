# Plot the trajectory of a localization topic for downstream visualization
# overlay with satellite imagery. The plot image should be a curve on a
# transparent background with nothing else in the image.

import argparse
import os
from typing import List
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


# TODO: automate rotation with a static start->endpoints angles

# accessible palettes
# https://github.com/matplotlib/matplotlib/issues/9460
# https://github.com/scverse/scanpy/issues/387#issuecomment-444803441
palette = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#279e68",
    "red": "#d62728",
    "purple": "#aa40fc",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#b5bd61",
    "cyan": "#17becf",
}

# https://matplotlib.org/stable/api/markers_api.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
plot_kwargs = {
    "6t": {
        "color": palette["purple"],
        "linestyle": "solid",
        "marker": "^",
        "markersize": 4,
        "markevery": (0.1, 0.1),
    },
    "3t": {
        "color": palette["orange"],
        "linestyle": (0, (5, 5, 1, 5)),  # longer dash of "dashdotted"
        "marker": None,
        "markevery": (0.1, 0.1),
    },
    "ganav": {
        "color": palette["cyan"],
        "linestyle": (0, (10, 3)),  # "long dash"
        "marker": None,
        "markevery": (0.1, 0.1),
    },
    "geometric": {
        "color": palette["red"],
        "linestyle": "dotted",
        "marker": None,
        "markevery": (0.1, 0.1),
    },
    "human": {
        "color": palette["green"],
        "linestyle": "solid",
        "marker": None,
        "markevery": (0.1, 0.1),
    },
}

display_keys = {
    "6t": "Retrained (ours)",
    "3t": "Base (ours)",
    "ganav": "GANav",
    "geometric": "Geometric",
    "human": "Human",
}

# colors = {
#     "6t": palette["purple"],
#     "3t": palette["orange"],
#     "ganav": palette["cyan"],
#     "geometric": palette["red"],
#     "human": palette["green"],
# }

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# Under section "Notes"
# fmt = {
#     "6t": "-.",
#     "3t": ":",
#     "ganav": "--",
#     "geometric": "^:",
#     "human": "-",
# }


def plot_trajectory(bagfile: Path, savedir: Path, pose_topic: str) -> None:
    date = bagfile.parent.parent.parent.stem
    location = bagfile.parent.parent.stem
    model_type = bagfile.parent.stem
    if model_type not in plot_kwargs.keys():
        logger.error(
            f"Skipping {bagfile}: model type '{model_type}' does not have an associated kwargs."
        )
        return

    bag = rosbag.Bag(bagfile)
    coords = np.empty((2, bag.get_message_count(pose_topic)), dtype=np.float64)

    if pose_topic == "/localization":
        loc_pose: Localization2DMsg
        for i, (_, loc_pose, _) in enumerate(bag.read_messages(pose_topic)):
            coords[0, i] = loc_pose.pose.x
            coords[1, i] = loc_pose.pose.y
    elif pose_topic == "/husky/inekf_estimation/pose":
        ekf_pose: PoseWithCovarianceStamped
        for i, (_, ekf_pose, _) in enumerate(bag.read_messages(pose_topic)):
            coords[0, i] = ekf_pose.pose.pose.position.x
            coords[1, i] = ekf_pose.pose.pose.position.y
    else:
        logger.error(f"Unrecognized topic: {pose_topic}")
        return

    # Translate the coordinates such that the starting point is (0, 0)
    coords -= coords[:, 0][:, np.newaxis]  # force a column vector

    # Rotate the coordinates such that the angular displacement is 0
    theta = np.arctan2(coords[1, -1] - coords[1, 0], coords[0, -1] - coords[0, 0])
    rotmat = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    coords = np.linalg.inv(rotmat) @ coords

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.plot(coords[0], coords[1], **plot_kwargs[model_type])

    savedir = (
        savedir
        / f"{date}-{location}"
        / model_type
        / ("loc" if pose_topic == "/localization" else "ekf")
    )

    os.makedirs(savedir, exist_ok=True)
    output_filename = savedir / (bagfile.stem + "-trajectory.png")
    fig.savefig(output_filename, transparent=True, dpi=200)
    logger.success(f"Saved trajectory image to {output_filename}")
    plt.close(fig)


# https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
def plot_legend() -> None:
    fig = pylab.figure()
    figlegend = pylab.figure()
    ax = fig.add_subplot(111)
    lines = []
    for key in plot_kwargs.keys():
        lines.append(
            ax.plot(range(10), range(10), **plot_kwargs[key], label=display_keys[key]),
        )
    legend = figlegend.legend(
        *ax.get_legend_handles_labels(), loc="center", framealpha=0.8
    )

    expand = [-5, -5, 5, 5]
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("legend.png", dpi=200, transparent=True, bbox_inches=bbox)

    # figlegend.savefig("legend.png")

    plt.close(fig)
    plt.close(figlegend)


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
        "--savedir",
        type=Path,
        default="images",
        help="Directory to save images in. Subdirectories will be created to reflect"
        "the source bagfile directory structure",
    )
    # argparser.add_argument(
    #     "--topic", type=str, default="/localization", help="Pose topic"
    # )
    args = argparser.parse_args()

    experiment_parent_dir: Path = args.experiment_parent_dir
    savedir: Path = args.savedir
    # topic: str = args.topic

    # for bagfile in bagfiles:
    #     plot_trajectory(bagfile, savedir, topic, color)

    for root, _, files in os.walk(experiment_parent_dir):
        for file in files:
            path = Path(root) / file
            # print(path)
            plot_trajectory(path, savedir, "/localization")
            plot_trajectory(path, savedir, "/husky/inekf_estimation/pose")

    plot_legend()
