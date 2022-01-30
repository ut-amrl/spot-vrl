"""Summarize the contents of a BDDF file in a format similar to `rosbag info`.

Invocation from the project's root directory:
    python3 -m spot_vrl.data.bddf_info [...]
"""

import argparse
from typing import Dict

import bosdyn.api.bddf_pb2
import bosdyn.bddf
import google.protobuf.timestamp_pb2

import spot_vrl.data.human_readable as hr


def summarize(filename: str, filter_series: bool = False) -> str:
    out: str = ""
    summary: Dict[str, str] = {}  # label -> value

    with open(filename, "rb") as fin:
        data_reader = bosdyn.bddf.DataReader(fin)

        summary["path"] = filename

        bddf_ver = data_reader.version
        summary[
            "bddf version"
        ] = f"{bddf_ver.major_version}.{bddf_ver.minor_version}.{bddf_ver.patch_level}"

        summary["robot species"] = data_reader.annotations["robot_species"]
        summary["robot version"] = data_reader.annotations["release_version"]

        # Mapped Fileindex
        message_counts: Dict[str, Dict[str, int]] = {}
        total_message_count = 0
        start_ts = hr.LocalTimestamp(40000000000)  # year 3237
        end_ts = hr.LocalTimestamp(0)
        msg_type_width = 0

        series_identifiers = data_reader.file_index.series_identifiers
        for index, series_identifier in enumerate(series_identifiers):
            series_type: str = series_identifier.series_type
            message_type: str = series_identifier.spec["bosdyn:message-type"]
            if not message_type:
                continue

            msg_type_width = max(msg_type_width, len(message_type))

            series_block_index: bosdyn.api.bddf_pb2.SeriesBlockIndex = (
                data_reader.series_block_index(index)
            )

            msg_count = len(series_block_index.block_entries)
            if msg_count != 0:
                total_message_count += len(series_block_index.block_entries)
                message_counts.setdefault(series_type, {})[message_type] = len(
                    series_block_index.block_entries
                )

            for block_entry in series_block_index.block_entries:
                ts: google.protobuf.timestamp_pb2.Timestamp = block_entry.timestamp
                lts = hr.LocalTimestamp(ts.seconds, ts.nanos)
                start_ts = min(start_ts, lts)
                end_ts = max(end_ts, lts)

        summary["duration"] = f"{start_ts.seconds_until(end_ts):.1f}s"
        summary["start"] = f"{start_ts.rfc1123()} ({start_ts.unix_time:.2f})"
        summary["end"] = f"{end_ts.rfc1123()} ({end_ts.unix_time:.2f})"
        summary["size"] = hr.filesize(filename)
        summary["messages"] = str(total_message_count)
        summary["types"] = "(filtered)" if filter_series else ""

        label_width = 0
        for label in summary.keys():
            # account for colon and space
            label_width = max(label_width, len(label) + 2)

        for label, value in summary.items():
            out += f"{label + ':':<{label_width}}{value}\n"

        for series_type in sorted(message_counts.keys()):
            if filter_series and series_type != "bosdyn:typed-message-channel":
                continue
            out += f"{'':4}{series_type}\n"
            for msg_type in sorted(message_counts[series_type].keys()):
                out += f"{'':<{label_width}}{msg_type:<{msg_type_width}} {message_counts[series_type][msg_type]:>4}\n"

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="File to open.")
    parser.add_argument(
        "--nofilter", action="store_true", help="Turns off series type filter."
    )

    options = parser.parse_args()
    print(summarize(options.filename, not options.nofilter), end="")


if __name__ == "__main__":
    main()
