#! /usr/bin/env python3

# Provides information about the size composition of topics within a rosbag.

# This file is intended to be symlinked to another directory and run as a
# standalone script. It should not be imported by any other python file or
# run with the python module system.

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import rosbag


def human_readable_size(bytes: float) -> str:
    suffixes = ["B", "KB", "MB", "GB"]

    log1000 = int(math.log10(bytes) // 3)
    return f"{bytes / (1000 ** log1000):.0f} {suffixes[log1000]}"


class RateLimit:
    def __init__(self, delta: float) -> None:
        self._delta = delta  # seconds
        self._last = time.time()

    def __call__(self) -> bool:
        now = time.time()
        if now - self._last > self._delta:
            self._last = now
            return True
        return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("bagfile", type=Path)
    argparser.add_argument(
        "--fast", action="store_true", help="Extrapolate one message"
    )
    args = argparser.parse_args()

    bagfile: Path = args.bagfile
    fast: bool = args.fast

    bag = rosbag.Bag(bagfile)
    total_msgs = bag.get_message_count()
    topic_size_sum: Dict[str, int] = defaultdict(int)
    topic_count: Dict[str, int] = defaultdict(int)

    for (
        topic,
        (
            msg_type,
            message_count,
            connections,
            frequency,
        ),
    ) in bag.get_type_and_topic_info().topics.items():
        topic_count[topic] = message_count

    if fast:
        for topic, count in topic_count.items():
            for (
                topic,
                (datatype, datablob, md5sum, (chunk_pos, offset), msg_type),
                stamp,
            ) in bag.read_messages(topics=topic, raw=True):
                topic_size_sum[topic] = len(datablob) * count
                break
    else:
        rate_limit = RateLimit(0.1)

        for (
            i,
            (
                topic,
                (datatype, datablob, md5sum, (chunk_pos, offset), msg_type),
                stamp,
            ),
        ) in enumerate(bag.read_messages(raw=True)):
            topic_size_sum[topic] += len(datablob)

            if rate_limit():
                print(
                    f"{i+1}/{total_msgs} ({i / total_msgs * 100:.0f}%)",
                    end="\r",
                    flush=True,
                )

    # sort in increasing order of total size
    topic_size_sum = dict(sorted(topic_size_sum.items(), key=lambda kv: kv[1]))

    topic_width = 40
    total_size_width = 8
    avg_size_width = 8
    for topic in topic_size_sum.keys():
        size_sum = topic_size_sum[topic]
        count = topic_count[topic]
        print(
            f"{topic:.<{topic_width}s}"
            f"{human_readable_size(size_sum):>{total_size_width}s}"
            f" ={human_readable_size(size_sum / count):>{avg_size_width}s}"
            f" x {count} msgs"
        )
