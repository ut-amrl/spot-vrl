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

from spot_vrl.imu_learning.datasets import ManualTripletDataset, ManualTripletHoldoutSet
from spot_vrl.imu_learning.losses import TripletLoss
from spot_vrl.imu_learning.network import (
    BaseEmbeddingNet,
    LstmEmbeddingNet,
    MlpEmbeddingNet,
    TripletNet,
)
from spot_vrl.imu_learning.trainer import EmbeddingGenerator, fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=("mlp", "lstm"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--steplr_step_size", type=int, default=10)
    parser.add_argument("--steplr_gamma", type=float, default=0.5)

    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    embedding_dim: int = args.embedding_dim
    model_type: str = args.model
    epochs: int = args.epochs
    margin: int = args.margin
    lr: float = args.lr
    batch_size: int = args.bs
    steplr_step_size: int = args.steplr_step_size
    steplr_gamma: int = args.steplr_gamma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data loaders
    triplet_dataset = ManualTripletDataset()
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
    loss_fn = TripletLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, steplr_step_size, gamma=steplr_gamma, last_epoch=-1
    )

    save_dir = ckpt_dir / f"{time.strftime('%m-%d-%H-%M-%S')}-{embedding_net.arch()}"
    os.makedirs(save_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(save_dir))  # type: ignore
    tb_writer.add_text("model", model_type)  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    tb_writer.add_text("steplr_step_size", str(steplr_step_size))  # type: ignore
    tb_writer.add_text("steplr_gamma", str(steplr_gamma))  # type: ignore

    embedder = EmbeddingGenerator(
        device, triplet_dataset, ManualTripletHoldoutSet(), tb_writer
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
