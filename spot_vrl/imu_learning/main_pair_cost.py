import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data.dataset
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spot_vrl.imu_learning.datasets import (
    PairCostTrainingDataset,
    TripletHoldoutDataset,
)
from spot_vrl.imu_learning.losses import MarginRankingLoss
from spot_vrl.imu_learning.network import (
    CostNet,
    FullPairCostNet,
    MlpEmbeddingNet,
)
from spot_vrl.imu_learning.cost_trainer import EmbeddingGenerator, fit


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument(
        "--encoder-model",
        type=Path,
        required=True,
        help="Path to saved encoder model.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    embedding_dim: int = args.embedding_dim
    encoder_path: Path = args.encoder_model
    dataset_dir: Path = args.dataset_dir
    continue_from: Path = args.continue_from
    start_epoch: int = 0
    epochs: int = args.epochs
    margin: float = args.margin
    lr: float = args.lr
    batch_size: int = args.bs
    comment: str = args.comment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset_spec: Path = dataset_dir / "train.json"
    holdout_dataset_spec: Path = dataset_dir / "holdout.json"

    if not train_dataset_spec.exists():
        logger.critical(
            f"Dataset specification {dataset_dir}/train.json does not exist."
        )
        sys.exit(1)
    elif not holdout_dataset_spec.exists():
        logger.critical(
            f"Dataset specification {dataset_dir}/holdout.json does not exist."
        )
        sys.exit(1)

    # Set up data loaders
    logger.info("Loading training data")
    cost_dataset = PairCostTrainingDataset(train_dataset_spec)
    train_size = int(len(cost_dataset) * 0.75)
    train_set, test_set = torch.utils.data.dataset.random_split(
        cost_dataset, (train_size, len(cost_dataset) - train_size)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    logger.info("Finished loading training data")

    # Set up the network and training parameters
    embedding_net = MlpEmbeddingNet(
        cost_dataset.triplet_dataset[0][0].shape, embedding_dim
    )
    embedding_net.load_state_dict(
        torch.load(encoder_path, map_location=device),  # type: ignore
        strict=True,
    )
    embedding_net.requires_grad_(False)
    cost_net = CostNet(embedding_dim)

    model = FullPairCostNet(embedding_net, cost_net)

    if continue_from is not None and os.path.exists(continue_from):
        model = torch.jit.load(continue_from, map_location=device)  # type: ignore
        start_epoch = int(continue_from.stem.split("_")[-1]) + 1

    model = model.to(device)

    loss_fn = MarginRankingLoss(margin)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    save_dir = (
        encoder_path.parent
        / f"cost-{encoder_path.stem.split('_')[-1]}"
        / time.strftime("%m-%d-%H-%M-%S")
    )
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(save_dir), flush_secs=1)  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    if comment:
        tb_writer.add_text("comment", comment)  # type: ignore

    logger.info("Loading evaluation data")
    embedder = EmbeddingGenerator(
        device,
        cost_dataset.triplet_dataset,
        TripletHoldoutDataset().init_from_json(holdout_dataset_spec),
        tb_writer,
    )
    logger.info("Finished loading evaluation data")

    fit(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        start_epoch,
        epochs,
        device,
        save_dir,
        tb_writer,
        embedder,
    )


if __name__ == "__main__":
    main()
