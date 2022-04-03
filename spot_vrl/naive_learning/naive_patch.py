import argparse
import enum
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.api.robot_id_pb2 import RobotIdResponse
from bosdyn.bddf import DataReader, ProtobufReader

from spot_vrl.data import ImuData, SpotImage, proto_to_numpy
from spot_vrl.homography import camera_transform, perspective_transform


# TODO(eyang): use values from robot states
BODY_HEIGHT_EST = 0.48938  # meters


class ImageWithText:
    def __init__(self, img: npt.NDArray[np.uint8]) -> None:
        self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.y = 0

    def add_line(self, text: str, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.25
        thickness = 3

        text_size = cv2.getTextSize(text, face, scale, thickness)[0]
        self.img = cv2.putText(
            self.img,
            text,
            (0, self.y + text_size[1]),
            face,
            scale,
            color,
            thickness=thickness,
        )
        self.y += text_size[1] + 5


class Terrain(enum.Enum):
    UNKNOWN = 0
    CONCRETE = 1
    GRASS = 2
    CONCRETE_TO_GRASS = 3
    GRASS_TO_CONCRETE = 4


class Datum:
    def __init__(
        self,
        ts: float,
        gp: float,
        image: npt.NDArray[np.uint8],
        odom_pose: npt.NDArray[np.float64],
    ) -> None:
        self.ts = ts
        """Timestamp (s)"""

        self.gp = gp
        """Mean ground depth (m) since the last data point"""

        self.image = image
        """Top-down image"""

        self.odom_pose = odom_pose

        self.rel_pose = np.identity(4)
        """The pose of this image relative to the previous image in the sequence
        as an Affine3D matrix."""


class SensorData:
    """Helper class for storing timestamped sensor data."""

    def __init__(self, filename: str) -> None:
        self.data: List[Datum] = []

        self._datapath: Path = Path(filename)
        self._data_reader = DataReader(None, filename)
        self._proto_reader = ProtobufReader(self._data_reader)

        self._start_ts: float = 0
        """Start timestamp reference for the data file."""

        self._init_start_ts()
        self._init_data()

    def _init_start_ts(self) -> None:
        series_index: int = self._proto_reader.series_index(
            "bosdyn.api.RobotIdResponse"
        )
        series_block_index: SeriesBlockIndex = self._data_reader.series_block_index(
            series_index
        )
        num_msgs = len(series_block_index.block_entries)
        assert num_msgs == 1

        _, ts, _ = self._proto_reader.get_message(series_index, RobotIdResponse, 0)
        self._start_ts = float(ts) * 1e-9

    def _init_data(self) -> None:
        assert self._start_ts != 0, "Initialize self._start_ts first."

        #
        # First read all of the depth values from robot states
        #

        # mapping of timestamps (s) to mean depth estimates for the timestamp
        depths: Dict[float, float] = {}

        imu = ImuData(self._datapath)
        depth_ts = imu.timestamp_sec - self._start_ts

        for t, d in zip(depth_ts, imu.foot_depth_mean):
            depths[t] = d

        #
        # Read in images and average the ground depth vals between image timestamps.
        # Depth values collected in the interval between images are attributed to the
        #   image at the start of the interval.
        #
        series_index = self._proto_reader.series_index("bosdyn.api.GetImageResponse")
        series_block_index = self._data_reader.series_block_index(series_index)
        num_msgs = len(series_block_index.block_entries)

        ground_tform_body = camera_transform.affine3d(
            [0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST]
        )

        for msg_idx in range(num_msgs):
            _, ts, response = self._proto_reader.get_message(
                series_index, GetImageResponse, msg_idx
            )
            ts = float(ts) * 1e-9 - self._start_ts

            images: List[SpotImage] = []
            for image_response in response.image_responses:
                image = SpotImage(image_response)
                if image.frame_name.startswith("front"):
                    images.append(image)
                    # image_save_path = (
                    #     Path("images")
                    #     / self._datapath.stem
                    #     / f"{msg_idx:03d}-{image.frame_name}.png"
                    # )
                    # cv2.imwrite(str(image_save_path), image.decoded_image())

            fused = perspective_transform.TopDown(images, ground_tform_body)

            tf_tree = response.image_responses[0].shot.transforms_snapshot
            body_tform_odom = proto_to_numpy.body_tform_frame(tf_tree, "odom")
            odom_tform_body = np.linalg.inv(body_tform_odom)

            self.data.append(Datum(ts, -1, fused.get_view(200), odom_tform_body))
            if msg_idx > 0:
                self.data[-1].rel_pose = (
                    np.linalg.inv(self.data[-2].odom_pose) @ self.data[-1].odom_pose
                )

        depth_it = iter(depths.items())
        depth_entry = next(depth_it)

        for i in range(len(self.data) - 1):
            ts = self.data[i].ts
            next_ts = self.data[i + 1].ts

            corr_depths: List[float] = []
            while depth_entry[0] < ts:
                try:
                    depth_entry = next(depth_it)
                except StopIteration:
                    depth_entry = (np.inf, 0)

            while depth_entry[0] >= ts and depth_entry[0] < next_ts:
                corr_depths.append(depth_entry[1])
                try:
                    depth_entry = next(depth_it)
                except StopIteration:
                    depth_entry = (np.inf, 0)

            mean_depth = 0.0
            if corr_depths:
                mean_depth = sum(corr_depths) / len(corr_depths)
            self.data[i].gp = mean_depth

    def process(self) -> None:
        for i in range(len(self.data) - 6):
            img_wrapper = ImageWithText(self.data[i].image)
            img_wrapper.add_line(f"ts: {self.data[i].ts:.3f}s")
            img_wrapper.add_line(f"fd: {self.data[i].gp * 100:.2f} cm")

            # resolution of 200 pixels / meter
            patch_size = 40 * 2
            l_tl_x = 213 * 2
            l_tl_y = 216 * 2
            l_br_x = l_tl_x + patch_size
            l_br_y = l_tl_y + patch_size

            r_tl_x = 272 * 2
            r_tl_y = 216 * 2
            r_br_x = r_tl_x + patch_size
            r_br_y = r_tl_y + patch_size

            print(f"{i:03d} ({self.data[i].ts:06.3f}s): ", end="")
            if self.data[i].gp * 100 < 0.01:
                print("invalid ground depth (<0.01 cm)")
                img_wrapper.add_line("invalid (<0.01 cm)")
            else:
                diff = self.data[i + 6].gp - self.data[i].gp
                zxy = Rotation.from_matrix(self.data[i].rel_pose[:3, :3]).as_euler(
                    "zxy", degrees=True
                )
                yaw = zxy[0]

                print(f"rotation ({yaw:.2f}º)", end="")

                if abs(yaw) >= 0.5 or abs(diff) * 100 > 1.4:
                    img_wrapper.add_line("invalid (transition)")
                    print()
                else:
                    if self.data[i].gp * 100 < 1.5:
                        img_wrapper.add_line("concrete")
                        print(" concrete")
                        patch_save_path = Path(
                            f"images/patches3/concrete/{self._datapath.stem}/{i:03d}"
                        )
                    else:
                        img_wrapper.add_line("grass")
                        print(" grass")
                        patch_save_path = Path(
                            f"images/patches3/grass/{self._datapath.stem}/{i:03d}"
                        )

                    img_wrapper.img = cv2.rectangle(
                        img_wrapper.img, (l_tl_x, l_tl_y), (l_br_x, l_br_y), (0, 255, 0)
                    )
                    img_wrapper.img = cv2.rectangle(
                        img_wrapper.img, (r_tl_x, r_tl_y), (r_br_x, r_br_y), (0, 255, 0)
                    )

                    os.makedirs(f"{patch_save_path}-L", exist_ok=True)
                    os.makedirs(f"{patch_save_path}-R", exist_ok=True)

                    left_patch = self.data[i].image[l_tl_y:l_br_y, l_tl_x:l_br_x]
                    right_patch = self.data[i].image[r_tl_y:r_br_y, r_tl_x:r_br_x]
                    cv2.imwrite(
                        f"{patch_save_path}-L/{i:03d}.png",
                        left_patch,
                    )
                    cv2.imwrite(
                        f"{patch_save_path}-R/{i:03d}.png",
                        right_patch,
                    )

                    colors = [
                        (255, 255, 0),
                        (0, 255, 255),
                        (255, 0, 255),
                    ]
                    future_steps = 3
                    future_pose = np.identity(4)
                    for j in range(future_steps):
                        if (i + j + 1 + 6) < len(self.data):
                            diff = self.data[i + j + 1 + 6].gp - self.data[i + j + 1].gp
                        else:
                            diff = 0

                        zxy = Rotation.from_matrix(
                            self.data[i + j + 1].rel_pose[:3, :3]
                        ).as_euler("zxy", degrees=True)
                        yaw = zxy[0]

                        if abs(yaw) >= 0.5 or abs(diff) * 100 > 1.4:
                            break

                        future_pose = self.data[i + j + 1].rel_pose @ future_pose

                        # adjust accordingly for dpi
                        fut_x, fut_y = (future_pose[:2, 3] * 200).astype(int)

                        # image and world axes are flipped
                        img_wrapper.img = cv2.rectangle(
                            img_wrapper.img,
                            (l_tl_x - fut_y, l_tl_y - fut_x),
                            (l_br_x - fut_y, l_br_y - fut_x),
                            colors[j % 3],
                        )
                        img_wrapper.img = cv2.rectangle(
                            img_wrapper.img,
                            (r_tl_x - fut_y, r_tl_y - fut_x),
                            (r_br_x - fut_y, r_br_y - fut_x),
                            colors[j % 3],
                        )

                        left_patch = self.data[i].image[
                            l_tl_y - fut_x : l_br_y - fut_x,
                            l_tl_x - fut_y : l_br_x - fut_y,
                        ]
                        right_patch = self.data[i].image[
                            r_tl_y - fut_x : r_br_y - fut_x,
                            r_tl_x - fut_y : r_br_x - fut_y,
                        ]

                        patch_save_path = patch_save_path.parent / f"{i +j + 1:03d}"
                        os.makedirs(f"{patch_save_path}-L", exist_ok=True)
                        os.makedirs(f"{patch_save_path}-R", exist_ok=True)
                        cv2.imwrite(
                            f"{patch_save_path}-L/{i:03d}.png",
                            left_patch,
                        )
                        cv2.imwrite(
                            f"{patch_save_path}-R/{i:03d}.png",
                            right_patch,
                        )

            image_save_path = Path("images") / self._datapath.stem / f"{i:03d}.png"
            cv2.imwrite(str(image_save_path), img_wrapper.img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    data = SensorData(options.filename)
    data.process()


if __name__ == "__main__":
    main()
