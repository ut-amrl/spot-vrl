import warnings

warnings.warn(
    "Homography transformation code has been ported to C++ to run during"
    " data collection (https://github.com/ut-amrl/local_rgb_map)."
    " This package is no longer maintained.",
    category=DeprecationWarning,
    stacklevel=2,
)
