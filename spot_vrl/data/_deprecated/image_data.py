import abc
import sys
from pathlib import Path
from typing import ClassVar, Dict, Final, Iterator, List, Tuple, Union, overload

import cv2
import numpy as np
import numpy.typing as npt
import simplejpeg

# from bosdyn.api import image_pb2
# from bosdyn.api.bddf_pb2 import SeriesBlockIndex
# from bosdyn.api.image_pb2 import GetImageResponse
# from bosdyn.bddf import DataReader, ProtobufReader
from loguru import logger
from scipy.spatial.transform import Rotation

if sys.platform == "linux":
    import rospy
    import rosbag
    from sensor_msgs.msg import CompressedImage

from spot_vrl.data._deprecated import ros_to_numpy
from spot_vrl.data._deprecated import proto_to_numpy
from spot_vrl.homography._deprecated import camera_transform

import warnings

warnings.warn(
    "The data pipeline has been fully migrated to ROS."
    " The Boston Dynamics Data Format (.bddf) compatibility interfaces"
    " in this module are no longer maintained.",
    category=DeprecationWarning,
    stacklevel=2,
)


class CameraImage(abc.ABC):
    """Base class for single images and associated camera metadata necessary
    for transforms."""

    @abc.abstractmethod
    def decoded_image(self) -> npt.NDArray[np.uint8]:
        """Decodes the raw data buffer as an image.

        The number of channels is dependent on the camera type:
            - Spot Camera: 1 channel (squeezed)
            - Kinect Camera: 3 channels (BGR order)

        Returns:
            npt.NDArray[np.uint8]: An image matrix of size
                (self.height, self.width) or (self.height, self.width, 3)
        """

    @abc.abstractmethod
    def decoded_image_ground_plane(self) -> npt.NDArray[np.uint8]:
        """Removes the sky from the image.

        Pixels with the value 0 in the original image (which are generally quite
        rare) are set to 1. Sky pixels are then "cleared" by setting their
        values to 0.

        Returns:
            npt.NDArray[np.uint8]: An image matrix of size
                (self.height, self.width) or (self.height, self.width, 3)
                containing the visible ground plane.
        """

    @property
    @abc.abstractmethod
    def height(self) -> int:
        """The height of this image."""

    @property
    @abc.abstractmethod
    def width(self) -> int:
        """The width of this image."""

    @property
    @abc.abstractmethod
    def channels(self) -> int:
        """The number of channels in this image."""

    @property
    @abc.abstractmethod
    def frame_name(self) -> str:
        """The name of the frame for this camera."""

    @property
    @abc.abstractmethod
    def body_tform_camera(self) -> npt.NDArray[np.float64]:
        """The 4x4 Affine3D pose of this camera in the body frame."""

    @property
    @abc.abstractmethod
    def body_height(self) -> np.float64:
        """The height of the body frame above the ground, used for
        homography."""

    @property
    def ground_tform_camera(self) -> npt.NDArray[np.float64]:
        """The 4x4 Affine3D pose of this camera relative to the ground
        below the body frame."""
        tform = np.copy(self.body_tform_camera)
        tform[2, 3] += self.body_height
        return tform

    @property
    @abc.abstractmethod
    def intrinsic_matrix(self) -> npt.NDArray[np.float64]:
        """The 3x3 intrinsic matrix of this camera."""

    @property
    @abc.abstractmethod
    def timestamp(self) -> np.float64:
        """The acquisition timestamp of this image."""


class SpotImage(CameraImage):
    _sky_masks: ClassVar[Dict[str, npt.NDArray[np.bool_]]] = {}
    """Cache for the image mask of the sky for each camera.

    Usage of this cache assumes the following are static:
      - Image Sizes
      - Camera-to-body transforms
      - The ground is always flat
    """

    def __init__(self, image_response) -> None:
        image_capture = image_response.shot
        image_source = image_response.source
        image = image_capture.image

        assert image_response.status == image_pb2.ImageResponse.Status.STATUS_OK
        assert (
            image_source.image_type == image_pb2.ImageSource.ImageType.IMAGE_TYPE_VISUAL
        )
        assert image.format == image_pb2.Image.Format.FORMAT_JPEG
        assert (
            image.pixel_format == image_pb2.Image.PixelFormat.PIXEL_FORMAT_GREYSCALE_U8
        )

        self._frame_name: str = image_capture.frame_name_image_sensor
        self._body_tform_camera = proto_to_numpy.body_tform_frame(
            image_capture.transforms_snapshot, self.frame_name
        )
        self._intrinsic_matrix = proto_to_numpy.camera_intrinsic_matrix(image_source)
        self._width: int = image.cols
        self._height: int = image.rows
        self._imgbuf: npt.NDArray[np.uint8] = np.frombuffer(image.data, dtype=np.uint8)
        self._ts = proto_to_numpy.timestamp_float64(image_capture.acquisition_time)

    def decoded_image(self) -> npt.NDArray[np.uint8]:
        """Decodes the raw data buffer as an image.

        Assumes the image is grayscale.

        Returns:
            npt.NDArray[np.uint8]: A 2D matrix of size (self.height, self.width)
                containing a single-channel image.
        """
        img: npt.NDArray[np.uint8] = cv2.imdecode(self._imgbuf, cv2.IMREAD_UNCHANGED)
        assert img.shape == (self.height, self.width)
        return img

    def _sky_mask(self) -> npt.NDArray[np.bool_]:
        """Calculates the image mask of the sky for the camera corresponding to
        this image.

        Returns:
            npt.NDArray[np.bool_]: A 2D matrix of size (self.height, self.width)
                containing a boolean bitmask where True represents the sky.
        """
        if self.frame_name not in self._sky_masks:
            # Generate a 2xN matrix of all integer image coordinates [x, y]
            image_coords = np.indices((self.width, self.height))
            image_coords = np.moveaxis(image_coords, 0, -1)
            image_coords = image_coords.reshape(self.width * self.height, 2)
            image_coords = image_coords.transpose()

            rays = camera_transform.camera_rays(
                image_coords, self.body_tform_camera, self.intrinsic_matrix
            )

            # Mark the indices of the rays that point above the horizon
            is_sky: npt.NDArray[np.bool_] = rays[2] >= 0
            # Convert flat array to (y, x) image coords
            is_sky = is_sky.reshape(self.width, self.height).T

            assert is_sky.shape == (self.height, self.width)
            self._sky_masks[self.frame_name] = is_sky

        return self._sky_masks[self.frame_name]

    def decoded_image_ground_plane(self) -> npt.NDArray[np.uint8]:
        """Removes the sky from the image.

        Pixels with the value 0 in the original image (which are generally quite
        rare) are set to 1. Sky pixels are then "cleared" by setting their
        values to 0.

        Returns:
            npt.NDArray[np.uint8]: A 2D matrix of size (self.height, self.width)
                containing a single-channel image of the visible ground plane.
        """
        img = self.decoded_image()

        img[img == 0] = 1
        img[self._sky_mask()] = 0

        return img

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def channels(self) -> int:
        return 1

    @property
    def frame_name(self) -> str:
        return self._frame_name

    @property
    def body_tform_camera(self) -> npt.NDArray[np.float64]:
        return self._body_tform_camera

    @property
    def body_height(self) -> np.float64:
        return KinectDefaults.BODY_HEIGHT["spot"]

    @property
    def intrinsic_matrix(self) -> npt.NDArray[np.float64]:
        return self._intrinsic_matrix

    @property
    def timestamp(self) -> np.float64:
        return self._ts


class KinectImage(CameraImage):
    """Container for single images and associated transforms metadata from the Azure
    Kinect DK camera."""

    # Hardcoded values for components of the transform data. Intrinsic
    # information was generated from the k4a_ros repository using the 1536p
    # setting. Extrinsic information was estimated using a ruler and gyroscope.
    EXPECTED_HEIGHT: Final[int] = 720
    EXPECTED_WIDTH: Final[int] = 1280

    # TODO: same structure as SpotImage, can put this in CameraImage
    _sky_masks: ClassVar[Dict[str, npt.NDArray[np.bool_]]] = {}

    def __init__(
        self,
        compressed_image: CompressedImage,
        intrinsic_matrix: npt.NDArray[np.float64],
        body_tform_camera: npt.NDArray[np.float64],
        body_height: np.float64,
    ) -> None:
        self._ts = np.float64(compressed_image.header.stamp.to_sec())
        self._imgbuf = np.frombuffer(compressed_image.data, dtype=np.uint8)

        self._intrinsic_matrix = intrinsic_matrix
        self._body_tform_camera = body_tform_camera
        self._body_height = body_height

    def decoded_image(self) -> npt.NDArray[np.uint8]:
        img: npt.NDArray[np.uint8] = simplejpeg.decode_jpeg(
            self._imgbuf,
            colorspace="BGR",
            min_height=self.EXPECTED_HEIGHT,
            min_width=self.EXPECTED_WIDTH,
        )
        if img.shape != (self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH, 3):
            logger.critical(
                f"Decoded image dimensions {img.shape} do not match "
                f"expected {(self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH, 3)}. "
                "Cannot proceed."
            )
            raise ValueError
        return img

    def _sky_mask(self) -> npt.NDArray[np.bool_]:
        """Calculates the image mask of the sky for the camera corresponding to
        this image.

        Returns:
            npt.NDArray[np.bool_]: A 2D matrix of size (self.height, self.width)
                containing a boolean bitmask where True represents the sky.
        """
        if self.frame_name not in self._sky_masks:
            # Generate a 2xN matrix of all integer image coordinates [x, y]
            image_coords = np.indices((self.width, self.height))
            image_coords = np.moveaxis(image_coords, 0, -1)
            image_coords = image_coords.reshape(self.width * self.height, 2)
            image_coords = image_coords.transpose()

            rays = camera_transform.camera_rays(
                image_coords, self.body_tform_camera, self.intrinsic_matrix
            )
            # Mark the indices of the rays that point above the horizon
            is_sky: npt.NDArray[np.bool_] = rays[2] >= 0
            # Convert flat array to (y, x) image coords
            is_sky = is_sky.reshape(self.width, self.height).T

            assert is_sky.shape == (self.height, self.width)
            self._sky_masks[self.frame_name] = is_sky

        return self._sky_masks[self.frame_name]

    def decoded_image_ground_plane(self) -> npt.NDArray[np.uint8]:
        """Removes the sky from the image.

        Pixels with the value 0 in the original image (which are generally quite
        rare) are set to 1. Sky pixels are then "cleared" by setting their
        values to 0.

        Returns:
            npt.NDArray[np.uint8]: A 3D matrix of size (self.height, self.width, 3)
                containing a BGR image of the visible ground plane.
        """
        img = self.decoded_image()
        img[img == 0] = 1
        img[self._sky_mask()] = 0
        return img

    @property
    def height(self) -> int:
        return self.EXPECTED_HEIGHT

    @property
    def width(self) -> int:
        return self.EXPECTED_WIDTH

    @property
    def channels(self) -> int:
        return 3

    @property
    def frame_name(self) -> str:
        return "kinect"

    @property
    def body_tform_camera(self) -> npt.NDArray[np.float64]:
        return self._body_tform_camera

    @property
    def body_height(self) -> np.float64:
        return self._body_height

    @property
    def intrinsic_matrix(self) -> npt.NDArray[np.float64]:
        return self._intrinsic_matrix

    @property
    def timestamp(self) -> np.float64:
        return self._ts


class ImageData(abc.ABC):
    """Iterable storage container for visual Spot data.

    The return type for all indexing operations is a tuple containing:
        - [0] np.float64: A unix timestamp.
        - [1] List[CameraImage]: A list of images corresponding to the timestamp.
    """

    def __init__(self, filename: Union[str, Path], lazy: bool = False) -> None:
        self._lazy = lazy
        self._path = Path(filename)

        # If `self._lazy` is True, these will remain empty.
        self._timestamps: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._images: List[List[CameraImage]] = []

    @classmethod
    def factory(cls, filename: Union[str, Path], lazy: bool = False) -> "ImageData":
        """Dispatch construction to subclass based on extension."""
        path = Path(filename)
        if path.suffix == ".bddf":
            return SpotImageData(filename, lazy)
        elif path.suffix == ".bag":
            return KinectImageData(filename)
        else:
            logger.error(f"Unrecognized file format ({path.suffix}).")
            raise ValueError

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @overload
    def __getitem__(self, idx: int) -> Tuple[np.float64, List[CameraImage]]:
        """
        Returns:
            Tuple[np.float64, List[CameraImage]]: A tuple containing the unix
                timestamp of the response message and a list of CameraImage
                objects.
        """

    @overload
    def __getitem__(
        self, idx: slice
    ) -> Tuple[npt.NDArray[np.float64], List[List[CameraImage]]]:
        """
        Returns:
            Tuple[npt.NDArray[np.float64], List[List[CameraImage]]]:
                Vectors containing unix timestamps of response messages
                and lists of CameraImage objects.

        Raises:
            RuntimeError:
                The class is set to lazy mode.
        """

    @abc.abstractmethod
    def __getitem__(self, idx):  # type: ignore
        ...

    class _Iterator(Iterator[Tuple[np.float64, List[CameraImage]]]):
        def __init__(self, container: "ImageData") -> None:
            self.idx = 0
            self.container = container

        def __iter__(self) -> Iterator[Tuple[np.float64, List[CameraImage]]]:
            return self

        def __next__(self) -> Tuple[np.float64, List[CameraImage]]:
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
    ) -> Tuple[npt.NDArray[np.float64], List[List[CameraImage]]]:
        """Queries a time range.

        Args:
            start (float): The start of the time range (unix seconds, inclusive).
            end (float): The end of the tne range (unix seconds, exclusive).

        Returns:
            Tuple[npt.NDArray[np.float64], List[List[CameraImage]]]:
                Vectors containing unix timestamps of response messages and
                lists of CameraImage objects corresponding to the time range.

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


class SpotImageData(ImageData):
    """Iterable storage container for visual Spot data.

    The return type for all indexing operations is a tuple containing:
        - [0] np.float64: A unix timestamp.
        - [1] List[CameraImage]: A list of images corresponding to the timestamp.

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
        super().__init__(filename, lazy)

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

    def _read_index(self, idx: int) -> Tuple[np.float64, List[CameraImage]]:
        """Reads and deconstructs a GetImageResponse from the BDDF file.

        Args:
            idx (int): The index of the GetImageResponse to read. Supports
                negative indices.

        Returns:
            Tuple[np.float64, List[CameraImage]]: A tuple containing the unix
                timestamp of the response message and a list of CameraImage
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
        images: List[CameraImage] = []
        for image_response in response.image_responses:
            images.append(SpotImage(image_response))
        return ts_sec, images

    def __len__(self) -> int:
        return self._num_msgs

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


class KinectDefaults:
    """Hardcoded properties for Kinect cameras. Intrinsic information was
    generated from the k4a_ros repository using the 1536p setting. Extrinsic
    information was estimated using a ruler.
    """

    BODY_HEIGHT: ClassVar[Dict[str, np.float64]] = {
        "spot": np.float64(0.48938),  # TODO: hardcoded value, try to use GPE frame?
        "jackal": np.float64(0),
        "unknown": np.float64(0),
    }

    INTRINSIC_MATRIX: ClassVar[Dict[str, npt.NDArray[np.float64]]] = {
        "spot": np.array(
            [
                [984.822632, 0, 1020.586365],
                [0, 984.730103, 780.647827],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ),
        "jackal": np.array(
            [
                [983.322571, 0, 1021.098450],
                [0, 983.123108, 775.020630],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ),
        "unknown": np.zeros((3, 3)),
    }

    BODY_TRANS_KINECT: ClassVar[Dict[str, npt.NDArray[np.float64]]] = {
        "spot": np.array([0.35, 0, 0.245], dtype=np.float64),
        "jackal": np.array([0.127, 0, 0.559], dtype=np.float64),
        "unknown": np.zeros(3),
    }

    BODY_ROT_KINECT_FALLBACK: ClassVar[npt.NDArray[np.float64]] = Rotation.from_euler(
        "XYZ", (0, 20, 0), degrees=True
    ).as_matrix()

    @staticmethod
    def detect_robot(bag: rosbag.Bag) -> str:
        """Tries to determine which robot the specified bagfile came from by
        searching for robot-specific topics.

        These topic lists will likely need to change over time.
        """

        spot_topics = [
            "/spot/odometry/twist",
            "/spot/status/battery_states",
            "/spot/status/feet",
        ]
        jackal_topics = ["/jackal_velocity_controller/odom"]

        if bag.get_message_count(spot_topics) > 0:
            return "spot"
        elif bag.get_message_count(jackal_topics) > 0:
            return "jackal"
        else:
            logger.error(
                f"Could not determine robot using bagfile topics ({bag.filename})."
            )
            return "unknown"


class KinectImageData(ImageData):
    def __init__(self, filename: Union[str, Path]) -> None:
        super().__init__(filename, False)

        bag = rosbag.Bag(str(self._path))
        robot_name = KinectDefaults.detect_robot(bag)

        body_rot_kinect: npt.NDArray[np.float64]
        if bag.get_message_count("/kinect_imu") > 0:
            imu_readings = []
            for _, msg, _ in bag.read_messages("/kinect_imu"):
                imu_readings.append(msg)
                if len(imu_readings) == 100:
                    break

            body_rot_kinect = ros_to_numpy.est_kinect_rot(imu_readings)
            r = Rotation.from_matrix(body_rot_kinect).as_euler("XYZ", degrees=True)
            logger.debug(f"kinect rotation estimate:\n{filename}: {r}")
        else:
            logger.warning(
                "Bagfile does not contain /kinect_imu msgs. Using default fallback of 20º pitch."
            )
            body_rot_kinect = KinectDefaults.BODY_ROT_KINECT_FALLBACK

        kinect_rot_camera = Rotation.from_euler(
            "XYZ", (-90, 90, 0), degrees=True
        ).as_matrix()

        body_tform_camera = np.identity(4)
        body_tform_camera[:3, 3] = KinectDefaults.BODY_TRANS_KINECT[robot_name]
        body_tform_camera[:3, :3] = body_rot_kinect @ kinect_rot_camera

        msg: CompressedImage
        ts: rospy.Time  # hmm does this match the KinectImage time?
        for _, msg, ts in bag.read_messages(topics="/camera/rgb/image_raw/compressed"):
            self._images.append(
                [
                    KinectImage(
                        msg,
                        KinectDefaults.INTRINSIC_MATRIX[robot_name],
                        body_tform_camera,
                        KinectDefaults.BODY_HEIGHT[robot_name],
                    )
                ]
            )

        self._timestamps = np.array(
            [img_list[0].timestamp for img_list in self._images], dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx):  # type: ignore
        return self._timestamps[idx], self._images[idx]
