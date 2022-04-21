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

from spot_vrl.imu_learning.datasets import (
    SingleTerrainDataset,
    TripletHoldoutDataset,
    TripletTrainingDataset,
)
from spot_vrl.imu_learning.network import (
    BaseEmbeddingNet,
    LstmEmbeddingNet,
    MlpEmbeddingNet,
    TripletNet,
)
from spot_vrl.imu_learning.trainer import EmbeddingGenerator, fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=("mlp", "lstm"))
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("0.5", "0.5-speedway-holdout", "1.0", "1.5"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument(
        "--window-size", type=float, default=SingleTerrainDataset.window_size
    )
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    embedding_dim: int = args.embedding_dim
    model_type: str = args.model
    dataset_dir: Path = Path("imu_datasets") / args.dataset
    epochs: int = args.epochs
    margin: int = args.margin
    lr: float = args.lr
    batch_size: int = args.bs
    window_size: int = args.window_size
    comment: str = args.comment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data loaders
    SingleTerrainDataset.set_global_window_size(window_size)
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
    embedding_net: BaseEmbeddingNet
    if model_type == "mlp":
        embedding_net = MlpEmbeddingNet(triplet_dataset[0][0].shape, embedding_dim)
    elif model_type == "lstm":
        embedding_net = LstmEmbeddingNet(triplet_dataset[0][0].shape, embedding_dim)
    else:
        logger.error(f"Unknown model type: '{model_type}")
        sys.exit(1)

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

    save_dir = ckpt_dir / f"{time.strftime('%m-%d-%H-%M-%S')}-{embedding_net.arch()}"
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(save_dir), flush_secs=10)  # type: ignore
    tb_writer.add_text("model", model_type)  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    tb_writer.add_text("window size", str(window_size))  # type: ignore
    if comment:
        tb_writer.add_text("comment", comment)  # type: ignore

    embedder = EmbeddingGenerator(
        device,
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
