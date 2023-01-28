# Visual Learning Pipeline as of 2023-01-28

## Background

This repository contains code for two different visual-inertial preference
learning projects. My branches (working name: PATURN) are "main" and anything
prefixed by "elvin/". All other branches should be Haresh's (working name:
NATURL).

The training code started out as modifications to Kavan's preference learning
code. A lot of code contained hacky object-oriented compatibility patches to
work with both Boston Dynamics Data Format data files and rosbags -- most of
this code has since been moved to `_deprecated` subpackages. The main training
code has been updated to only ROS data, but various utility scripts have not
been updated.

My code is constantly changing and I cannot guarantee any of this will work in
two days or outside of the very Spot-specific workflow I've been using, but
hopefully it shouldn't be too hard to adapt.

## Dependencies

Python package dependencies are specified in `pyproject.toml` (and `poetry.lock`
if you want to use poetry).

`ffmpeg` is required in various places to create videos. It's already installed
on `robofleet`, `robovision` and `robodata`.

ROS is required to parse rosbags and create datasets, so they can't be created
on `robofleet` anymore. However, datasets can be created on `robodata` and
afterwards loaded on `robofleet` for training.

## Rough Architecture Overview

`spot_vrl/data` contains lower-level utilities for parsing rosbags and "data
container" classes that allow for iteration and time range queries. If you're
using Spot, you probably won't need to touch these, but adaptations to
`sensor_data.py` may be needed for other robots.

---

`spot_vrl/visual_learning` contains everything related to actually training the
visual models.

- `datasets.py` contains PyTorch dataset creation, serialization, and
  deserialization functions.
  - The `SingleTerrainDataset` class is an internal class for extracting patches
    from single, continuous robot trajectories.
    - Patches are extracted by scanning forward (Haresh's patch extraction works
      by scanning backward). As an example, when processing the 100th image, the
      algorithm extracts patches in the 100th image at the poses of the robot at
      the timestamps of the 101th, 102nd, 103rd... images relative to the pose
      at the timestamp of the 100th image. The image patches themselves are
      stored in a massive contiguous tensor (for fast serialization). The image
      index relationships are stored in `patch_idx_lookup`, which maps an image
      index number to a list of _previous_ image index numbers in which a patch
      was extracted. Following the above example, keys 101, 102, and 103 would
      all contain `[100]`, meaning that a patch for the robot's pose at the
      timestamp of the 101st image (and 102, 103, etc) was extracted from the
      viewpoint of the robot at the pose of the 100th image timestamp. Thus, the
      lowest value in the list will generally be the blurriest patch since it
      was extracted from the oldest viewpoint.
  - The `BaseTripletDataset` class stores concatenated `SingleTerrainDatasets`
    organized by terrains (referred to as "categories" in code).
    - Turning of video generation in `_pll_load_or_create` will speed up dataset
      creation substantially.
    - ## IMPORTANT
      Dataset creation/loading is parallelized, and PyTorch Tensors use shared
      memory to speed up data transfers between processes. If your datasets are
      very large, they program may crash because the system will run out of
      shared memory. You can change the internal representation of the massive
      patch stack in `SingleTerrainDataset` to a numpy array (and convert
      patches to Tensors in `__getitem__`) to circumvent this, but loading
      datasets will be much slower.
    - It also specifies how full (anchor, positive, negative) triplets are
      sampled for training.
- `network.py` contains network architectures.
- `main_representation.py` is the entry point for representation learning.
  - `trainer.py` contains train/validation steps
- `main_pair_cost.py` is the entry point for cost function learning.

---

`spot_vrl/utils` contains:

- `video_writer.py` for creating videos.
- `parallel.py` for wrangling `multiprocessing`, `tqdm`, and `loguru` to work together nicely.

Both of these are used primarily in `spot_vrl/visual_learning/datasets.py`

## Workflow

### Data Collection

Without any modifications, the current visual learning pipeline requires (read:
breaks without) the following topics:

- BEV images (`/bev/single/compressed`) for image patches.
- Odometry data (`/tf`) for forward patch extraction.
  - I parse the `/tf` tree -- I think this is a relic from the old BDDF files
    that only had odom information in kinematic trees. `/odom` should also work,
    but you'll need to modify `spot_vrl/data/sensor_data.py`.
- `/joint_states`
  - Shared code dependency with inertial learning.
- `/odom`
  - Shared code dependency with inertial learning.
- `/spot/odometry/twist`
  - Shared code dependency with inertial learning.
- `/spot/status/battery_states`
  - Shared code dependency with inertial learning.
- `/spot/status/feet`
  - Shared code dependency with inertial learning.

Also recommended:
- `/camera/rgb/image_raw/compressed` for visualization.

### Dataset Specification

Datasets are specified using JSON files. I think it's probably just easiest to
look at some examples in `datasets/`. Training needs both `train.json` and
`holdout.json`. I've symlinked `/robodata/eyang/data` as `data` in my project
directory.

The `orderings` field is a bit confusing -- I intend to change the label
representation at some point, but the way it works now is:

- A label of `-1` means that `first` is preferable to the `second`
- A label of `0` means that `first` and `second` are equally preferable
- A label of `1` means that `first` is less preferable to `second`.

### Representation Learning

An example of the general invocation from the root of the project directory:

```shell
python -m spot_vrl.visual_learning.main_representation --dataset-dir datasets/v2-initial --embedding-dim 8 --ckpt-dir visual-models/v2-initial
```

Run with the `-h` flag or look at `main_representation.py` to get a list of
optional arguments with defaults.

---

Somewhat important:

The default implementation of `torch.utils.tensorboard._embedding.make_mat` does
not buffer its writes and tries to flush after every line or something. This is
extremely slow on networked filesystems. I recommend copying the code snippet I
put at top of `trainer.py` into your virtualenv or just turning off embeddings
in `trainer.fit`

### Cost Learning

An example of the general invocation from the root of the project directory:

```shell
python -m spot_vrl.visual_learning.main_pair_cost --dataset-dir datasets/v2-initial --embedding-dim 8 --triplet-model visual-models/v2-initial/mm-dd-HH-MM-SS/trained_epoch_79.pth --ckpt-dir visual-models/v2-initial/mm-dd-HH-MM-SS/cost/
```

`embedding-dim` must be the same as the one used to train the visual encoder.
