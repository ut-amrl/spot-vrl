import argparse
from pathlib import Path

import torch

from spot_vrl.visual_learning.network import (
    CostNet,
    EmbeddingNet,
    FullPairCostNet,
    TripletNet,
)


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
    return cost_model


@torch.no_grad()  # type: ignore
def make_cost_vid(filename: Path, cost_net: CostNet) -> None:
    ...


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
        required=True,
        help="Path to BDDF file to visualize.",
    )

    args = parser.parse_args()

    embedding_dim: int = args.embedding_dim
    cost_model_path: Path = args.cost_model
    datafile_path: Path = args.datafile

    cost_model = load_cost_model(cost_model_path, embedding_dim)
    make_cost_vid(datafile_path, cost_model.cost_net)


if __name__ == "__main__":
    main()
