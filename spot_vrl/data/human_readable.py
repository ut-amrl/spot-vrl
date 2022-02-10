"""Human-readable representations of common numerical types."""

import math
import time


def filesize(bytes: int) -> str:
    """Returns a quantity of bytes in a human-readable format.

    Uses binary (base 1024) units, e.g. 12 B, 34.5 KiB, 678.9 MiB

    Args:
        bytes: The number of bytes.
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    base = 1024.0

    size = float(bytes)
    unit_idx = 0
    while size > base and unit_idx + 1 < len(units):
        size /= base
        unit_idx += 1

    precision = 0 if unit_idx == 0 else 1
    return f"{size:.{precision}f} {units[unit_idx]}"


class LocalTimestamp:
    """Converts and formats a UNIX timestamp into the local time zone.

    The documentation for Go has examples of various date/time formats.
    https://pkg.go.dev/time#pkg-constants
    """
    def __init__(self, seconds: int, nanos: int = 0) -> None:
        self._unix_time: float = float(seconds) + float(nanos) * 1e-9
        self._local_time = time.localtime(math.floor(self._unix_time + 0.5))

    @property
    def unix_time(self) -> float:
        return self._unix_time

    def seconds_until(self, other: "LocalTimestamp") -> float:
        """Returns the difference in seconds between the provided timestamp
        parameter and this timestamp.
        """
        return other._unix_time - self._unix_time

    def rfc1123(self) -> str:
        """Returns this timestamp in the same format as:

        `Mon, 02 Jan 2006 15:04:05 MST`
        """
        return time.strftime("%a, %d %b %Y %H:%M:%S %Z", self._local_time)

    def __lt__(self, other: "LocalTimestamp") -> bool:
        return self._unix_time < other._unix_time

    def __gt__(self, other: "LocalTimestamp") -> bool:
        return self._unix_time > other._unix_time
