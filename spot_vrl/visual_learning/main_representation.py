import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data.dataset
from loguru import logger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spot_vrl.visual_learning.datasets import (
    TripletHoldoutDataset,
    TripletTrainingDataset,
)
from spot_vrl.visual_learning.network import (
    EmbeddingNet,
    TripletNet,
)
from spot_vrl.visual_learning.trainer import EmbeddingGenerator, fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    embedding_dim: int = args.embedding_dim
    dataset_dir: Path = args.dataset_dir
    epochs: int = args.epochs
    margin: int = args.margin
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

    triplet_dataset = TripletTrainingDataset().init_from_json(train_dataset_spec)
    logger.debug(f"Training dataset size: {len(triplet_dataset)}")

    train_size = int(len(triplet_dataset) * 0.75)
    train_set, test_set = torch.utils.data.dataset.random_split(
        triplet_dataset, (train_size, len(triplet_dataset) - train_size)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger.info("Finished loading data")

    # Set up the network and training parameters
    embedding_net = EmbeddingNet(embedding_dim)

    model = TripletNet(embedding_net)
    model = model.to(device)
    loss_fn = torch.nn.TripletMarginLoss(margin=margin, swap=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
        verbose=True,
    )

    save_dir = ckpt_dir / f"{time.strftime('%m-%d-%H-%M-%S')}"
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(save_dir))  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    if comment:
        tb_writer.add_text("comment", comment)  # type: ignore

    logger.info("Loading Evaluation Data")
    embedder = EmbeddingGenerator(
        device,
        batch_size,
        triplet_dataset,
        TripletHoldoutDataset().init_from_json(holdout_dataset_spec),
        tb_writer,
    )
    logger.info("Finished Loading Evaluation Data")

    fit(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        epochs,
        device,
        save_dir,
        tb_writer,
        embedder=embedder,
    )


if __name__ == "__main__":
    main()
