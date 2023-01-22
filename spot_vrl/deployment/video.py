# Generate a video from a rosbag, showing the camera feed, timestamp,
# distance traveled (odom, ekf).


# expected fps of camera feed: 15 fps


# 1280 x 720 video with approximate frame
"""
ts rel to 0:                       raw camera feed
\int d, odom:                      bev camera feed
\int d, ekf:                       costmap camera feed

"""

import argparse
import bisect
import os
import os.path
from pathlib import Path
from typing import Generic, List, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
import rosbag
import tqdm
from geometry_msgs.msg import PoseWithCovarianceStamped
from loguru import logger
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage

import spot_vrl.utils.parallel as parallel
from spot_vrl.utils.video_writer import ImageWithText, VideoWriter

_T = TypeVar("_T")


class TopicData(Generic[_T]):
    def __init__(self) -> None:
        self._ts: List[float] = []

        # Store all the messages. Memory inefficient, but the rosbag API doesn't
        # have random access
        self._msgs: List[_T] = []

    def get_ge(self, ts_start: float) -> _T:
        idx = bisect.bisect_left(self._ts, ts_start)
        return self._msgs[idx]  # caller handles IndexError

    def get_ge_lt(self, ts_start: float, ts_end: float) -> List[_T]:
        idx_start = bisect.bisect_left(self._ts, ts_start)
        idx_end = bisect.bisect_left(self._ts, ts_end)
        return self._msgs[idx_start:idx_end]


class BagData:
    def __init__(self, bagfile: Path) -> None:
        self._raw_imgs = TopicData[CompressedImage]()
        self._bev_imgs = TopicData[CompressedImage]()
        self._cost_imgs = TopicData[CompressedImage]()
        self._odom_poses = TopicData[Odometry]()
        self._ekf_poses = TopicData[PoseWithCovarianceStamped]()
        self._topics = {
            "/camera/rgb/image_raw/compressed": self._raw_imgs,
            "/bev/single/compressed": self._bev_imgs,
            "/vis_image/compressed": self._cost_imgs,
            "/odom": self._odom_poses,
            "/husky/inekf_estimation/pose": self._ekf_poses,
        }

        # Use the bagfile timestamps instead of header timestamps because some
        # topics (e.g. the cost image from the navigation node) are not
        # published with meaningful timestamps
        bag = rosbag.Bag(bagfile)
        for topic, msg, ts in bag.read_messages(list(self._topics.keys())):
            self._topics[topic]._ts.append(ts.to_sec())
            self._topics[topic]._msgs.append(msg)

        # for topic, data in self._topics.items():
        #     logger.debug(f"{topic}: {len(data._ts)} msgs")

    def get_frame(self, i: int) -> npt.NDArray[np.uint8]:
        if i < 0 or i >= len(self):
            logger.error("Frame index out of bounds")
            return np.empty(0, dtype=np.uint8)

        # hardcoded size will dictate the size of all other components
        frame_width = 1280
        frame_height = 720
        frame = np.zeros((frame_height, frame_width, 3), np.uint8)

        # Base everything on the raw image timestamp since the logical sequence
        # of events starts with a raw image, then the bev image, then the cost image
        raw_image_ts = self._raw_imgs._ts[i]

        # Stack the bev image on top of the raw image on the right side of the frame
        raw_image = cv2.imdecode(
            np.frombuffer(self._raw_imgs._msgs[i].data, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        raw_image_width = 576
        raw_image_height = raw_image_width * 3 // 4
        raw_image = cv2.resize(raw_image, (raw_image_width, raw_image_height))
        frame[-raw_image_height:, -raw_image_width:] = raw_image

        try:
            # approximately a 2:1 image (closer to 1.94:1), but who can tell the
            # difference anyway
            bev_image = cv2.imdecode(
                np.frombuffer(self._bev_imgs.get_ge(raw_image_ts).data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            bev_image_width = raw_image_width
            bev_image_height = bev_image_width // 2
            bev_image = cv2.resize(bev_image, (bev_image_width, bev_image_height))
            # draw a line at the bottom center of the image to show where the robot is
            bev_image[
                -50:, bev_image_width // 2 - 2 : bev_image_width // 2 + 2, 2
            ] = 255
            frame[:bev_image_height, -bev_image_width:] = bev_image
        except IndexError:
            pass

        # Write the relative timestamp, odom dist, and ekf dist since start at
        # the top left of the frame
        text_wrapper = ImageWithText(frame)
        text_wrapper.add_line(
            f"ts: {raw_image_ts - self._raw_imgs._ts[0]:.3f}", (255, 255, 255)
        )

        # could cache the value for the prev idx, but get something up and
        # running now
        try:
            odom_msgs = self._odom_poses.get_ge_lt(self._raw_imgs._ts[0], raw_image_ts)
            odom_dist = 0.0
            for i in range(1, len(odom_msgs)):
                dx = (
                    odom_msgs[i].pose.pose.position.x
                    - odom_msgs[i - 1].pose.pose.position.x
                )
                dy = (
                    odom_msgs[i].pose.pose.position.y
                    - odom_msgs[i - 1].pose.pose.position.y
                )
                odom_dist += np.sqrt(dx * dx + dy * dy)
            text_wrapper.add_line(f"odom: {odom_dist:.3f}", (255, 255, 255))
        except IndexError:
            pass

        # could cache the value for the prev idx, but get something up and
        # running now
        try:
            ekf_msgs = self._ekf_poses.get_ge_lt(self._raw_imgs._ts[0], raw_image_ts)
            ekf_dist = 0.0
            for i in range(1, len(ekf_msgs)):
                dx = (
                    ekf_msgs[i].pose.pose.position.x
                    - ekf_msgs[i - 1].pose.pose.position.x
                )
                dy = (
                    ekf_msgs[i].pose.pose.position.y
                    - ekf_msgs[i - 1].pose.pose.position.y
                )
                ekf_dist += np.sqrt(dx * dx + dy * dy)
            text_wrapper.add_line(f"ekf: {ekf_dist:.3f}", (255, 255, 255))
        except IndexError:
            pass

        return frame

    def __len__(self) -> int:
        return len(self._raw_imgs._ts)


def generate_video(bagfile: Path) -> None:
    # expect some form of "/**/date/location/model_type/*.bag"
    date = bagfile.parent.parent.parent.stem
    location = bagfile.parent.parent.stem
    model_type = bagfile.parent.stem

    video_path = (
        Path("images") / f"{date}-{location}" / model_type / f"{bagfile.stem}.mp4"
    )
    os.makedirs(video_path.parent, exist_ok=True)

    data = BagData(bagfile)
    video_writer = VideoWriter(video_path, 15)
    for i in tqdm.trange(
        0, len(data), position=parallel.tqdm_position(), dynamic_ncols=True, leave=False
    ):
        frame = data.get_frame(i)
        video_writer.add_frame(frame)

    video_writer.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "experiment_parent_dir",
        type=Path,
        nargs="+",
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
    args = argparser.parse_args()

    experiment_parent_dir: List[Path] = args.experiment_parent_dir

    paths = []

    # ok so it might actually be a file
    for dir in experiment_parent_dir:
        if os.path.isfile(dir):
            paths.append(dir)
        else:
            for root, _, files in os.walk(dir):
                for file in files:
                    if file.endswith(".bag"):
                        path = Path(root) / file
                        paths.append(path)

    parallel.fork_join(generate_video, paths, 6)
