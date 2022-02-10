import argparse
from typing import Tuple

import numpy as np
import numpy.typing as npt

from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.robot_state_pb2 import RobotStateResponse
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.bddf import DataReader, ProtobufReader


def timestamps(
    filename: str,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    state = []
    img = []

    data_reader = DataReader(None, filename)
    proto_reader = ProtobufReader(data_reader)

    series_idx: int = proto_reader.series_index("bosdyn.api.RobotStateResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_idx)
    num_msgs = len(series_block_index.block_entries)

    for msg_idx in range(num_msgs):
        timestamp_nsec: int
        _, timestamp_nsec, _ = proto_reader.get_message(
            series_idx, RobotStateResponse, msg_idx
        )
        state.append(timestamp_nsec * 1.0 * 1e-9)

    series_idx: int = proto_reader.series_index("bosdyn.api.GetImageResponse")
    series_block_index: SeriesBlockIndex = data_reader.series_block_index(series_idx)
    num_msgs = len(series_block_index.block_entries)

    for msg_idx in range(num_msgs):
        _, timestamp_nsec, _ = proto_reader.get_message(
            series_idx, GetImageResponse, msg_idx
        )
        img.append(timestamp_nsec * 1.0 * 1e-9)

    return np.array(state), np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")

    options = parser.parse_args()
    state, img = timestamps(options.filename)

    state_period = np.zeros(len(state) - 1)
    img_period = np.zeros(len(img) - 1)

    for i in range(1, len(state)):
        state_period[i - 1] = state[i] - state[i - 1]

    for i in range(1, len(img)):
        img_period[i - 1] = img[i] - img[i - 1]

    state_freq = 1 / state_period
    img_freq = 1 / img_period

    print(f"state: {state_freq.mean()} / {state_freq.std()}")
    print(f"img: {img_freq.mean()} / {img_freq.std()}")


if __name__ == "__main__":
    main()
