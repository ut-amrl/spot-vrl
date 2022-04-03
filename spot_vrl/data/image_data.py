from pathlib import Path
from typing import Iterator, List, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
from bosdyn.api.bddf_pb2 import SeriesBlockIndex
from bosdyn.api.image_pb2 import GetImageResponse
from bosdyn.bddf import DataReader, ProtobufReader
from loguru import logger

from spot_vrl.homography.proto_to_numpy import SpotImage


class ImageData:
    """Iterable storage container for visual Spot data.

    The return type for all indexing operations is a tuple containing:
        - [0] np.float64: A unix timestamp.
        - [1] List[SpotImage]: A list of images corresponding to the timestamp.

    The returned timestamp is the timestamp stored in the GetImageResponse
    message from the robot. The response timestamp was chosen because the actual
    capture timestamps of the individual images may be different from one
    another. The response timestamp will be at least several milliseconds
    greater than the actual image capture timestamps.

    This class has an option to load images lazily to reduce memory use.
    Indexing with slices is not supported when lazy loading.

    This class assumes the GetImageResponses stored in the BDDF file are sorted
    by ascending timestamp. Our data collection stack should enforce ordering,
    but arbitrary BDDF files may not necessarily be ordered.
    """

    def __init__(self, filename: Union[str, Path], lazy: bool = False) -> None:
        """
        Args:
            filename (str | Path): Path to a BDDF file.
            lazy (bool): Whether to load images lazily to reduce memory use.
        """
        self._lazy = lazy

        # If `self._lazy` is True, these will remain empty.
        self._timestamps: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._images: List[List[SpotImage]] = []

        self._path = Path(filename)
        self._data_reader = DataReader(None, str(self._path))
        self._proto_reader = ProtobufReader(self._data_reader)

        # Store various low-level BDDF properties.
        self._series_index: int = self._proto_reader.series_index(
            "bosdyn.api.GetImageResponse"
        )
        self._series_block_index: SeriesBlockIndex = (
            self._data_reader.series_block_index(self._series_index)
        )
        self._num_msgs = len(self._series_block_index.block_entries)

        if not self._lazy:
            self._timestamps = np.empty(self._num_msgs, dtype=np.float64)
            for msg_idx in range(len(self)):
                ts_sec, images = self._read_index(msg_idx)
                self._timestamps[msg_idx] = ts_sec
                self._images.append(images)

    def _read_index(self, idx: int) -> Tuple[np.float64, List[SpotImage]]:
        """Reads and deconstructs a GetImageResponse from the BDDF file.

        Args:
            idx (int): The index of the GetImageResponse to read. Supports
                negative indices.

        Returns:
            Tuple[np.float64, List[SpotImage]]: A tuple containing the unix
                timestamp of the response message and a list of SpotImage
                objects.

        Raises:
            IndexError: The specified index is out of bounds.
        """
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError

        _, _, response = self._proto_reader.get_message(
            self._series_index, GetImageResponse, idx
        )
        header_ts = response.header.response_timestamp

        ts_sec = float(header_ts.seconds) + header_ts.nanos * 1e-9
        images: List[SpotImage] = []
        for image_response in response.image_responses:
            images.append(SpotImage(image_response))
        return ts_sec, images

    def __len__(self) -> int:
        return self._num_msgs

    @overload
    def __getitem__(self, idx: int) -> Tuple[np.float64, List[SpotImage]]:
        """
        Returns:
            Tuple[np.float64, List[SpotImage]]: A tuple containing the unix
                timestamp of the response message and a list of SpotImage
                objects.
        """

    @overload
    def __getitem__(
        self, idx: slice
    ) -> Tuple[npt.NDArray[np.float64], List[List[SpotImage]]]:
        """
        Returns:
            Tuple[npt.NDArray[np.float64], List[List[SpotImage]]]:
                Vectors containing unix timestamps of response messages
                and lists of SpotImage objects.

        Raises:
            RuntimeError:
                The class is set to lazy mode.
        """

    def __getitem__(self, idx):  # type: ignore
        if self._lazy:
            if type(idx) == int:
                return self._read_index(idx)
            else:
                raise RuntimeError(
                    "Indexing with slices is not supported in lazy mode."
                )
        else:
            return self._timestamps[idx], self._images[idx]

    class _Iterator(Iterator[Tuple[np.float64, List[SpotImage]]]):
        def __init__(self, container: "ImageData") -> None:
            self.idx = 0
            self.container = container

        def __iter__(self) -> Iterator[Tuple[np.float64, List[SpotImage]]]:
            return self

        def __next__(self) -> Tuple[np.float64, List[SpotImage]]:
            if self.idx < len(self.container):
                ret_tuple = self.container[self.idx]
                self.idx += 1
                return ret_tuple
            else:
                raise StopIteration

    def __iter__(self) -> _Iterator:
        return self._Iterator(self)

    def query_time_range(
        self, start: float = 0, end: float = np.inf
    ) -> Tuple[npt.NDArray[np.float64], List[List[SpotImage]]]:
        """Queries a time range.

        Args:
            start (float): The start of the time range (unix seconds, inclusive).
            end (float): The end of the tne range (unix seconds, exclusive).

        Returns:
            Tuple[npt.NDArray[np.float64], List[List[SpotImage]]]:
                Vectors containing unix timestamps of response messages and
                lists of SpotImage objects corresponding to the time range.

        Raises:
            ValueError:
                The time range is invalid.
            RuntimeError:
                The class is in lazy mode.
        """
        if self._lazy:
            raise RuntimeError("Range query is not supported in lazy mode.")

        if start >= end:
            raise ValueError("start >= end")
        if start > self._timestamps[-1] + 1 or end < self._timestamps[0] - 1:
            logger.warning(
                f"Specified time range ({start}, {end}) and dataset {self._path} are disjoint."
            )

        start_i = int(np.searchsorted(self._timestamps, start))
        end_i = int(np.searchsorted(self._timestamps, end))

        if start_i == end_i:
            logger.warning("Returning empty arrays.")
        return self[start_i:end_i]
