import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger

from spot_vrl.data.image_data import ImageData, SpotImage
from spot_vrl.homography import camera_transform, perspective_transform
from spot_vrl.utils.video_writer import VideoWriter
from spot_vrl.visual_learning.network import (
    CostNet,
    EmbeddingNet,
    FullPairCostNet,
    TripletNet,
)

# TODO: fix bad coding practices
from spot_vrl.scripts.fuse_images import estimate_fps


def load_cost_model(path: Path, embedding_dim: int) -> FullPairCostNet:
    embedding_net = EmbeddingNet(embedding_dim)
    triplet_net = TripletNet(embedding_net)
    cost_net = CostNet(embedding_dim)

    cost_model = FullPairCostNet(triplet_net, cost_net)
    cost_model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu")),  # type: ignore
        strict=True,
    )
    cost_model.requires_grad_(False)
    cost_model.eval()
    return cost_model


@torch.no_grad()  # type: ignore
def make_cost_vid(filename: Path, cost_model: FullPairCostNet) -> None:
    imgdata = ImageData(filename, lazy=True)

    BODY_HEIGHT_EST = 0.48938  # meters
    GROUND_TFORM_BODY = camera_transform.affine3d([0, 0, 0, 1], [0, 0, BODY_HEIGHT_EST])

    fps = estimate_fps(imgdata)
    video_writer = VideoWriter(Path("images") / f"{filename.stem}-cost.mp4", fps=fps)

    count = 0

    images: List[SpotImage]
    for _, images in tqdm.tqdm(
        imgdata, desc="Processing Cost Video", total=len(imgdata), dynamic_ncols=True
    ):
        images = [image for image in images if "front" in image.frame_name]
        td = perspective_transform.TopDown(images, GROUND_TFORM_BODY)
        view = td.get_view(resolution=150)
        cost_view = np.zeros((*view.shape, 3), dtype=np.uint8)

        PATCH_SIZE = 60
        MAX_COST = 2.0
        for row in range(0, view.shape[0], PATCH_SIZE):
            patch_y = np.s_[row : min(view.shape[0], row + PATCH_SIZE)]
            for col in range(0, view.shape[1], PATCH_SIZE):
                patch_x = np.s_[col : min(view.shape[1], col + PATCH_SIZE)]
                patch = torch.from_numpy(view[patch_y, patch_x])[None, ...]

                # TODO: find actual min size limits based on kernel sizes
                if patch.shape[1] < 16 or patch.shape[2] < 16 or 0 in patch:
                    continue

                cost: float = cost_model.get_cost(patch).squeeze().item()

                # color map:
                # gradient:
                #   bright red = high cost
                #   dark = low cost
                red = cost / MAX_COST
                red = np.clip(red, 0.0, 1.0)

                cost_view[patch_y, patch_x, 2] = int(red * 255)

        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        view = np.vstack((view, cost_view))
        video_writer.add_frame(view)

        count += 1
        if count > 1000:
            break

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

    parser.add_argument("--embedding-dim", type=int, required=True)
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

    embedding_dim: int = args.embedding_dim
    cost_model_path: Path = args.cost_model
    datafile_path: Path = args.datafile

    cost_model = load_cost_model(cost_model_path, embedding_dim)
    make_cost_vid(datafile_path, cost_model)


if __name__ == "__main__":
    main()
