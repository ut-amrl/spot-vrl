import abc
from pathlib import Path
from typing import Iterator, List, Tuple, Union, overload

import numpy as np
import numpy.typing as npt
import simplejpeg
from loguru import logger

import rosbag
from sensor_msgs.msg import CompressedImage


class Image:
    """Container to lazily decode (JPEG) ROS CompressedImage messages."""

    def __init__(self, compressed_image: CompressedImage) -> None:
        self._timestamp = np.float64(compressed_image.header.stamp.to_sec())
        self._imgbuf = np.frombuffer(compressed_image.data, dtype=np.uint8)

    def decoded_image(self) -> npt.NDArray[np.uint8]:
        img: npt.NDArray[np.uint8] = simplejpeg.decode_jpeg(
            self._imgbuf,
            colorspace="RGB",
        )
        return img


class DecodedImage(Image):
    """For backwards-compatibility with legacy code."""

    def __init__(
        self, timestamp: np.float64, decoded_image: npt.NDArray[np.uint8]
    ) -> None:
        self._timestamp = timestamp
        self._decoded_image = decoded_image

    def decoded_image(self) -> npt.NDArray[np.uint8]:
        return self._decoded_image


class ImageSequence(abc.ABC):
    """Iterable storage container for image data.

    The return type for all indexing operations is a tuple containing:
        - [0] np.float64: A unix timestamp.
        - [1] np.NDArray[np.uint8]: An RGB image.
    """

    def __init__(self, filename: Union[str, Path]) -> None:
        self._path = Path(filename)
        self._timestamps: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._images: List[Image] = []

    def __len__(self) -> int:
        return len(self._images)

    @overload
    def __getitem__(self, idx: int) -> Tuple[np.float64, Image]:
        """
        Returns:
            Tuple[np.float64, Image]: A tuple containing a unix
                timestamp and an Image object.
        """

    @overload
    def __getitem__(self, idx: slice) -> Tuple[npt.NDArray[np.float64], List[Image]]:
        """
        Returns:
            Tuple[npt.NDArray[np.float64], List[Image]]:
                Vectors containing unix timestamps and Image objects.
        """

    def __getitem__(self, idx):  # type: ignore
        return self._timestamps[idx], self._images[idx]

    class _Iterator(Iterator[Tuple[np.float64, Image]]):
        def __init__(self, container: "ImageSequence") -> None:
            self.idx = 0
            self.container = container

        def __iter__(self) -> Iterator[Tuple[np.float64, Image]]:
            return self

        def __next__(self) -> Tuple[np.float64, Image]:
            if self.idx < len(self.container):
                ret_tuple = self.container[self.idx]
                self.idx += 1
                return ret_tuple
            else:
                raise StopIteration

    def __iter__(self) -> _Iterator:
        return self._Iterator(self)

    def query_time_range(
        self,
        start: Union[float, np.float_] = 0,
        end: Union[float, np.float_] = np.inf,
    ) -> Tuple[npt.NDArray[np.float64], List[Image]]:
        """Queries a time range.

        Args:
            start (float): The start of the time range (unix seconds, inclusive).
            end (float): The end of the tne range (unix seconds, exclusive).

        Returns:
            Tuple[npt.NDArray[np.float64], List[Image]]:
                Vectors containing unix timestamps of response messages and
                Image objects corresponding to the time range.

        Raises:
            ValueError:
                The time range is invalid.
        """
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


class KinectImageSequence(ImageSequence):
    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__(filename)

        bag = rosbag.Bag(str(filename))
        msg: CompressedImage
        for _, msg, _ in bag.read_messages("/camera/rgb/image_raw/compressed"):
            self._images.append(Image(msg))

        self._timestamps = np.array(
            [image._timestamp for image in self._images], dtype=np.float64
        )


class BEVImageSequence(ImageSequence):
    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__(filename)

        bag = rosbag.Bag(str(filename))
        if bag.get_message_count("/bev/single/compressed") != 0:
            msg: CompressedImage
            for _, msg, _ in bag.read_messages("/bev/single/compressed"):
                self._images.append(Image(msg))
        else:
            # fallback to old code and hope that it all still works
            from spot_vrl.data._deprecated.image_data import KinectImageData
            from spot_vrl.homography._deprecated.perspective_transform import TopDown

            image_data = KinectImageData(filename)

            for timestamp, list_with_a_single_image in image_data:
                bev_image = TopDown(list_with_a_single_image).get_view(150)
                # Deprecated code used BGR. Convert to RGB
                bev_image = bev_image[:, :, ::-1]
                self._images.append(DecodedImage(timestamp, bev_image))

        self._timestamps = np.array(
            [image._timestamp for image in self._images], dtype=np.float64
        )
