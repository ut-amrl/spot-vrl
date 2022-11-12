# Plot the trajectory of a localization topic for downstream visualization
# overlay with satellite imagery. The plot image should be a curve on a
# transparent background with nothing else in the image.

import argparse
import os
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import rosbag
from amrl_msgs.msg import Localization2DMsg
from loguru import logger


def plot_trajectory(bagfile: Path, loc_topic: str, color: str) -> None:
    # The [x, y] coordinates in Localization2DMsg match the axes convention used
    # by matplotlib. If the topic is changed to an Odometry message, the
    # coordinates will need to be rotated 90deg clockwise to plot correctly.
    x: "List[float]" = []
    y: "List[float]" = []

    msg: Localization2DMsg
    for _, msg, _ in rosbag.Bag(bagfile).read_messages(loc_topic):
        x.append(msg.pose.x)
        y.append(msg.pose.y)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.plot(x, y, color=color)

    output_filename = Path("images") / (bagfile.stem + "-trajectory.png")
    os.makedirs(output_filename.parent, exist_ok=True)
    fig.savefig(output_filename, transparent=True, dpi=200)
    logger.success(f"Saved trajectory image to {output_filename}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("bagfile", type=Path, help="Path to bagfile")
    argparser.add_argument(
        "--loc-topic", type=str, default="/localization", help="Localization topic"
    )
    argparser.add_argument(
        "--color",
        type=str,
        default="tab:blue",
        help="Matplotlib color string (https://matplotlib.org/stable/gallery/color/named_colors.html)",
    )
    args = argparser.parse_args()

    bagfile: Path = args.bagfile
    loc_topic: str = args.loc_topic
    color: str = args.color

    plot_trajectory(bagfile, loc_topic, color)
