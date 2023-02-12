import io
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import PIL
from PIL import Image as PIL_Image
from loguru import logger


class ImageWithText:
    """
    Helper class to add text lines to the top left corner of an RGB image.
    """

    def __init__(self, img: npt.NDArray[np.uint8]) -> None:
        self.img = img
        self._y = 0

    def add_line(self, text: str, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        """Adds a text line to the image and issues a virtual CRLF."""
        face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2

        text_size = cv2.getTextSize(text, face, scale, thickness)[0]
        self.img = cv2.putText(
            self.img,
            text,
            (0, self._y + text_size[1]),
            face,
            scale,
            color,
            thickness=thickness,
        )
        self._y += text_size[1] + 5


class VideoWriter:
    """
    Creates a video from image frames.

    This class is a replacement for OpenCV's VideoWriter class, which I've found
    encounters platform-specific errors (e.g. different fourcc codes are needed
    on different platforms).

    This class makes the following decisions for the user:
        - Backend: ffmpeg
        - Video encoding: x264
        - Video extension: .mp4
        - Chroma subsampling: yuv444 (doesn't require even dimensions)

    Videos are created by streaming images to an ffmpeg subprocess. The size of
    each frame is guessed from the first streamed image and is enforced
    thereafter.

    Example:
        >>> writer = VideoWriter()
        >>> for image in images:
        >>>     writer.add_frame(image)
        >>> writer.close()
    """

    def __init__(self, filename: Union[str, Path], fps: int = 8) -> None:
        """Constructs a VideoWriter.

        Args:
            filename (str | Path): The output file path.
            fps (int): The output video frame rate.

        Raises:
            FileNotFoundError: ffmpeg cannot be found, probably because it is
                either missing or not in $PATH
            BrokenPipeError: The ffmpeg subprocess' stdin handle could not be
                opened, thus preventing images from being streamed. This
                shouldn't ever happen, and if it does I don't know how to fix it
                other than trying again, rebooting, etc.
        """
        filename = Path(filename)
        if filename.suffix != ".mp4":
            filename = filename.parent / filename.stem / ".mp4"
        os.makedirs(filename.parent, exist_ok=True)
        self._filename = filename

        # fmt: off
        ffmpeg_args = [
            "ffmpeg",
            "-loglevel", "warning",
            "-y",
            "-f", "image2pipe",
            "-probesize", "32M",
            "-r", str(fps),  # input fps needs to be specified or frames may be dropped
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "23",
            "-vf", "format=yuv444p",
            "-r", str(fps),
            str(filename)
        ]
        # fmt: on

        self._shape: Optional[Tuple[int, ...]] = None
        """The expected shape of image frames. Note that C-style dimension
        ordering is used instead of the convention used for images."""

        self._ffmpeg = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        if self._ffmpeg.stdin is None:
            raise BrokenPipeError("Subprocess' stdin could not be opened.")

    def add_frame(self, frame: npt.NDArray[np.uint8]) -> None:
        """Add a frame to the video.

        Args:
            frame (npt.NDArray[np.uint8]): The unencoded frame as an array.

        Raises:
            ValueError: The shape of `frame` is invalid or does not match a
                previously added image.
            ValueError: The image could not be encoded.
            ChildProcessError: The ffmpeg process has terminated already.
        """
        if frame.ndim < 2 or frame.ndim > 3:
            raise ValueError(f"Expected 2 or 3 dimensions, got {frame.ndim}.")

        if self._shape is None:
            self._shape = frame.shape
        elif frame.shape != self._shape:
            raise ValueError(f"Expected array shape {self._shape}, got {frame.shape}.")

        if self._ffmpeg.returncode:
            raise ChildProcessError("The ffmpeg process has terminated.")

        # PIL is slightly faster than OpenCV due to not needing to convert RGB to BGR
        with io.BytesIO() as image_buffer:
            im = PIL_Image.fromarray(frame)
            im.save(image_buffer, format="BMP")

            assert self._ffmpeg.stdin  # redundant check, silence Optional[] check
            self._ffmpeg.stdin.write(image_buffer.getvalue())

        # # OpenCV expects the input to be in BGR order.
        # if frame.ndim == 3:
        #     frame = frame[:, :, ::-1]

        # img_buf: npt.NDArray[np.uint8]
        # success, img_buf = cv2.imencode(".bmp", frame)
        # if not success:
        #     raise ValueError("The image could not be encoded.")

        # assert self._ffmpeg.stdin  # redundant check, silence Optional[] check
        # self._ffmpeg.stdin.write(img_buf.tobytes())

    def close(self) -> None:
        """Closes the ffmpeg subprocess."""
        if self._ffmpeg.returncode is None:
            assert self._ffmpeg.stdin
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait()

            if self._ffmpeg.returncode == 0:
                logger.success(f"Saved video to {self._filename}")

    def __del__(self) -> None:
        """Closes the ffmpeg subprocess as a fallback."""
        if self._ffmpeg.returncode is None:
            logger.warning(
                "Closing ffmpeg subprocess at object destruction."
                " This should not be relied upon, use close() instead."
            )
            self.close()
