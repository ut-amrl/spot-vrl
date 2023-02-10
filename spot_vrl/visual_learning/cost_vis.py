import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
from loguru import logger

from spot_vrl.data.image_data import BEVImageSequence, Image
from spot_vrl.visual_learning.datasets import PATCH_SIZE, zero_pixel_ratio
from spot_vrl.utils.parallel import tqdm_position, fork_join
from spot_vrl.utils.video_writer import VideoWriter


def load_cost_model(path: Path) -> torch.jit.ScriptModule:
    cost_model: torch.jit.ScriptModule = torch.jit.load(
        path, map_location=torch.device("cpu")
    )  # type: ignore
    cost_model.eval()
    return cost_model


@torch.no_grad()  # type: ignore
def make_cost_vid(filename: Path, cost_model_path: Path) -> None:
    cost_model: torch.jit.ScriptModule = load_cost_model(cost_model_path)
    bev_image_data = BEVImageSequence(filename)

    fps = 15
    video_writer = VideoWriter(
        Path("images") / f"cost-eval/{filename.stem}.mp4", fps=fps
    )

    image: Image
    for _, image in tqdm.tqdm(
        bev_image_data,
        desc=f"Processing {filename.name}",
        total=len(bev_image_data),
        dynamic_ncols=True,
        position=tqdm_position(),
        leave=False,
    ):
        view = image.decoded_image()
        cost_view = np.zeros(view.shape, dtype=np.uint8)
        cost_view[:, :, 0] = 255

        MIN_COST = 0
        MAX_COST = 3.5
        for row in range(0, view.shape[0], PATCH_SIZE):
            patch_y = np.s_[row : min(view.shape[0], row + PATCH_SIZE)]

            # Compute each row as a batch to increase throughput
            patch_slices = []
            patch_imgs = []

            for col in range(0, view.shape[1], PATCH_SIZE):
                patch_x = np.s_[col : min(view.shape[1], col + PATCH_SIZE)]
                patch = view[patch_y, patch_x]

                # this changes RGB to BGR, if necessary (e.g. with old models)
                # patch = patch[:, :, ::-1]

                if (
                    patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
                    and zero_pixel_ratio(patch) < 0.5
                ):
                    patch_t = torch.from_numpy(patch.copy()).permute((2, 0, 1))[None, :]

                    patch_slices.append(patch_x)
                    patch_imgs.append(patch_t)

            if len(patch_imgs) == 0:
                continue

            costs = cost_model(torch.cat(patch_imgs))

            for i in range(len(patch_slices)):
                patch_x = patch_slices[i]

                cost = costs[i]
                # logger.debug(cost)

                # color map:
                # gradient:
                #   bright = high cost
                #   dark = low cost
                intensity = (cost - MIN_COST) / (MAX_COST - MIN_COST)
                intensity = np.clip(intensity, 0.0, 1.0)
                cost_view[patch_y, patch_x] = int(intensity * 255)

        view = np.vstack((view, cost_view))
        video_writer.add_frame(view)

    video_writer.close()


def main() -> None:
    """
    Load model

    for each fused image
        copy image (costmap image)
        for each horizontal patch row:
            for each vertical patch row:
                if 0 not in image:
                    feed patch into model
                    assign color to patch in costmap image

        stack images
        feed stacked image into videowriter

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cost-model",
        type=Path,
        required=True,
        help="Path to saved JIT CostNet model.",
    )
    parser.add_argument(
        "bagfiles",
        type=Path,
        nargs="+",
        help="Path to rosbag to visualize.",
    )

    args = parser.parse_args()

    cost_model_path: Path = args.cost_model
    bagfile_paths: List[Path] = args.bagfiles

    task_args = [(bagfile_path, cost_model_path) for bagfile_path in bagfile_paths]
    fork_join(make_cost_vid, task_args, n_proc=4)


if __name__ == "__main__":
    main()
