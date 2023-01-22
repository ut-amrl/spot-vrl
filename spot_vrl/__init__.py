import multiprocessing
import os
import warnings

import cv2

__version__ = "0.1.0"


def _get_omp_num_threads() -> int:
    try:
        count = int(os.environ["OMP_NUM_THREADS"])
    except (KeyError, ValueError):
        count = multiprocessing.cpu_count()
    return count


def _set_omp_num_threads(count: int) -> None:
    # OpenMP thread limit must be set before importing libraries that read the
    # environment variable.
    os.environ["OMP_NUM_THREADS"] = str(count)
    cv2.setNumThreads(count)


# On machines with very high core counts, overhead from library multithreading
# (NumPy, OpenCV) can slow programs down.
_set_omp_num_threads(min(8, _get_omp_num_threads()))

warnings.simplefilter("once", DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*rosbag.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*torch.*utils.*tensorboard.*",
)
