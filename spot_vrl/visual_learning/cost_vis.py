import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger

from spot_vrl.data.image_data import ImageData, SpotImage, CameraImage
from spot_vrl.homography._deprecated import perspective_transform
from spot_vrl.utils.video_writer import VideoWriter

# TODO: fix bad coding practices
from spot_vrl.scripts.fuse_images import estimate_fps


def load_cost_model(path: Path) -> torch.jit.ScriptModule:
    cost_model: torch.jit.ScriptModule = torch.jit.load(
        path, map_location=torch.device("cpu")
    )
    cost_model.eval()
    return cost_model


@torch.no_grad()  # type: ignore
def make_cost_vid(filename: Path, cost_model: torch.jit.ScriptModule) -> None:
    imgdata = ImageData.factory(filename, lazy=True)[::3]

    # fps = estimate_fps(imgdata)
    fps = 15
    video_writer = VideoWriter(Path("images") / f"{filename.stem}-cost.mp4", fps=fps)

    images: List[CameraImage]
    # for _, images in tqdm.tqdm(
    #     imgdata, desc="Processing Cost Video", total=len(imgdata), dynamic_ncols=True
    # ):
    for images in tqdm.tqdm(
        imgdata[1],
        desc="Processing Cost Video",
        total=len(imgdata[1]),
        dynamic_ncols=True,
    ):
        # images = [image for image in images if "front" in image.frame_name]
        td = perspective_transform.TopDown(images)
        view = td.get_view(resolution=150, horizon_dist=5.0)
        cost_view = np.zeros(view.shape, dtype=np.uint8)

        PATCH_SIZE = 60
        MIN_COST = 0
        MAX_COST = 2
        for row in range(0, view.shape[0], PATCH_SIZE):
            patch_y = np.s_[row : min(view.shape[0], row + PATCH_SIZE)]

            # Compute each row as a batch to increase throughput
            patch_slices = []
            patch_imgs = []

            for col in range(0, view.shape[1], PATCH_SIZE):
                patch_x = np.s_[col : min(view.shape[1], col + PATCH_SIZE)]
                patch = torch.from_numpy(view[patch_y, patch_x]).permute((2, 0, 1))[
                    None, :
                ]

                if patch.shape[2] < 60 or patch.shape[3] < 60 or 0 in patch:
                    continue

                patch_slices.append(patch_x)
                patch_imgs.append(patch)

            if len(patch_imgs) == 0:
                continue

            costs = cost_model(torch.cat(patch_imgs))

            for i in range(len(patch_slices)):
                patch_x = patch_slices[i]

                cost = costs[i]
                # logger.debug(cost)

                # color map:
                # gradient:
                #   bright red = high cost
                #   dark = low cost
                red = (cost - MIN_COST) / (MAX_COST - MIN_COST)
                red = np.clip(red, 0.0, 1.0)
                cost_view[patch_y, patch_x, 2] = int(red * 255)

        # view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
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
        help="Path to saved CostNet model.",
    )
    parser.add_argument(
        "datafile",
        type=Path,
        help="Path to BDDF file to visualize.",
    )

    args = parser.parse_args()

    cost_model_path: Path = args.cost_model
    datafile_path: Path = args.datafile

    cost_model = load_cost_model(cost_model_path)
    make_cost_vid(datafile_path, cost_model)


if __name__ == "__main__":
    main()
