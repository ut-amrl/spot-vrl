import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataset
from loguru import logger
from torch.optim import lr_scheduler

from spot_vrl.imu_learning.datasets import ManualTripletDataset
from spot_vrl.imu_learning.losses import TripletLoss
from spot_vrl.imu_learning.network import (
    BaseEmbeddingNet,
    LstmEmbeddingNet,
    MlpEmbeddingNet,
    TripletNet,
)
from spot_vrl.imu_learning.trainer import fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=("mlp", "lstm"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--steplr_step_size", type=int, default=10)
    parser.add_argument("--steplr_gamma", type=float, default=0.5)

    args = parser.parse_args()

    ckpt_dir: str = args.ckpt_dir
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
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

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

    os.makedirs(ckpt_dir, exist_ok=True)

    fit(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        epochs,
        device,
        ckpt_dir,
    )


if __name__ == "__main__":
    main()
