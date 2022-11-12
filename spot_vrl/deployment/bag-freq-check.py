#! /usr/bin/env python3

# Checks the average frequency of rosbag topics against a minimum expected
# value.

# This file is intended to be symlinked to another directory and run as a
# standalone script. It should not be imported by any other python file or
# run with the python module system.

import argparse
import sys
from typing import Dict, Tuple, Union
from pathlib import Path

import rosbag
from loguru import logger  # will need to be installed via pip


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("bagfile", type=Path)
    argparser.add_argument(
        "-q", action="store_true", help="Quieter output: do not print successes."
    )

    args = argparser.parse_args()
    bagfile: Path = args.bagfile
    quiet: bool = args.q

    logger.remove()
    logger.add(sys.stderr, format=" <lvl>{level:9s}</lvl> | <lvl>{message}</lvl>")

    min_freqs: Dict[Union[str, Tuple[str, ...]], float] = {
        # image observations
        "/bev/single/compressed": 14.9,
        "/camera/rgb/image_raw/compressed": 14.9,
        # lidar observations
        "/velodyne_2dscan_highbeams": 9.85,
        "/velodyne_points": 9.85,
        # proprioception
        "/joint_states": 20,
        "/kinect_imu": 100,
        "/vectornav/IMU": 39.5,
        "/odom": 20,
        "/spot/odometry/twist": 20,
        "/spot/status/feet": 20,
        # navigation debug topics
        "/ganav/cost_image/compressed": 10,
        "/ganav/cost_image_bev/compressed": 10,
        "/vis_image/compressed": 20,
        "/navigation/costmap/compressed": 20,
        "/navigation/cmd_vel": 30,
        "/visualization": 60,  # published by two topics, not the most accurate check
        (
            "/move_base_simple/goal",
            "/move_base_simple/goal_amrl",
        ): 0,  # check for at least one msg of either topic
        # navigation trajectory information
        "/husky/inekf_estimation/pose": 39.5,
        "/localization": 20,
    }

    bag = rosbag.Bag(bagfile)
    duration = bag.get_end_time() - bag.get_start_time()
    logger.info(f"Duration: {duration:.3f}s")

    width = 60
    for topic, min_hz in min_freqs.items():
        count = bag.get_message_count(topic)

        if count == 0:
            logger.error(f"{str(topic):{width}s} 0")
        else:
            hz = count / duration
            if hz < min_hz:
                logger.warning(f"{str(topic):{width}s} {hz:.2f} < {min_hz:.2f}")
            elif not quiet:
                logger.success(f"{str(topic):{width}s} {hz:.2f} >= {min_hz:.2f}")
