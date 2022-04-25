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
    SingleTerrainDataset,
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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("0.5-no-speedway", "0.5-with-speedway", "0.5-speedway-holdout"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--steplr-step-size", type=int, default=10)
    parser.add_argument("--steplr-gamma", type=float, default=0.5)
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    embedding_dim: int = args.embedding_dim
    dataset_dir: Path = Path("visual-datasets") / args.dataset
    epochs: int = args.epochs
    margin: int = args.margin
    lr: float = args.lr
    batch_size: int = args.bs
    steplr_step_size: int = args.steplr_step_size
    steplr_gamma: int = args.steplr_gamma
    comment: str = args.comment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data loaders
    triplet_dataset = TripletTrainingDataset().init_from_json(
        dataset_dir / "train.json"
    )
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
    scheduler = lr_scheduler.StepLR(
        optimizer, steplr_step_size, gamma=steplr_gamma, last_epoch=-1
    )

    save_dir = ckpt_dir / f"{time.strftime('%m-%d-%H-%M-%S')}"
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(save_dir))  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    tb_writer.add_text("steplr_step_size", str(steplr_step_size))  # type: ignore
    tb_writer.add_text("steplr_gamma", str(steplr_gamma))  # type: ignore
    if comment:
        tb_writer.add_text("comment", comment)  # type: ignore

    embedder = EmbeddingGenerator(
        device,
        batch_size,
        triplet_dataset,
        TripletHoldoutDataset().init_from_json(dataset_dir / "holdout.json"),
        tb_writer,
    )

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
