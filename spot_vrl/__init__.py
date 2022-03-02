import os
import multiprocessing
import typing


__version__ = "0.1.0"


def _get_omp_num_threads() -> int:
    count: typing.Union[int, str] = os.environ["OMP_NUM_THREADS"]
    if not count:
        count = multiprocessing.cpu_count()
    return int(count)


def _set_omp_num_threads(count: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(count)


# On machines with very high core counts, parallelism exploited by libraries can
# slow programs down due to shared state / communication overhead.
_set_omp_num_threads(min(8, _get_omp_num_threads()))
