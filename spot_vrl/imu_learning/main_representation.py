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
)
from spot_vrl.imu_learning.trainer import EmbeddingGenerator, fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=("mlp", "lstm"))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument(
        "--window-size", type=float, default=SingleTerrainDataset.window_size
    )
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()

    embedding_dim: int = args.embedding_dim
    model_type: str = args.model
    dataset_dir: Path = args.dataset_dir
    continue_from: Path = args.continue_from
    start_epoch: int = 0
    epochs: int = args.epochs
    margin: int = args.margin
    lr: float = args.lr
    batch_size: int = args.bs
    window_size: int = args.window_size
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

    SingleTerrainDataset.set_global_window_size(window_size)
    triplet_dataset = TripletTrainingDataset().init_from_json(train_dataset_spec)
    train_size = int(len(triplet_dataset) * 0.75)
    train_set, test_set = torch.utils.data.dataset.random_split(
        triplet_dataset, (train_size, len(triplet_dataset) - train_size)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger.info("Finished loading data")

    # Set up the network and training parameters
    model: BaseEmbeddingNet
    if model_type == "mlp":
        model = MlpEmbeddingNet(triplet_dataset[0][0].shape, embedding_dim)
    elif model_type == "lstm":
        model = LstmEmbeddingNet(triplet_dataset[0][0].shape, embedding_dim)
    else:
        logger.error(f"Unknown model type: '{model_type}")
        sys.exit(1)

    if continue_from is not None and os.path.exists(continue_from):
        model.load_state_dict(
            torch.load(continue_from, map_location=device),  # type: ignore
            strict=True,
        )
        model.requires_grad_(True)
        start_epoch = int(continue_from.stem.split("_")[-1]) + 1

    model = model.to(device)
    loss_fn = torch.nn.TripletMarginLoss(margin=margin, swap=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
        verbose=True,
    )

    save_dir = (
        Path("imu-models")
        / dataset_dir.name
        / f"{time.strftime('%m-%d-%H-%M-%S')}-{model.arch()}"
    )
    os.makedirs(save_dir, exist_ok=True)
    if continue_from is not None and os.path.exists(continue_from):
        os.symlink(
            os.path.relpath(continue_from, save_dir), save_dir / continue_from.name
        )

    tb_writer = SummaryWriter(log_dir=str(save_dir), flush_secs=10)  # type: ignore
    tb_writer.add_text("model", model_type)  # type: ignore
    tb_writer.add_text("margin", str(margin))  # type: ignore
    tb_writer.add_text("lr", str(lr))  # type: ignore
    tb_writer.add_text("bs", str(batch_size))  # type: ignore
    tb_writer.add_text("window size", str(window_size))  # type: ignore
    if comment:
        tb_writer.add_text("comment", comment)  # type: ignore

    logger.info("Loading evaluation data")
    embedder = EmbeddingGenerator(
        device,
        triplet_dataset,
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
        scheduler,
        start_epoch,
        epochs,
        device,
        save_dir,
        tb_writer,
        embedder=embedder,
    )


if __name__ == "__main__":
    main()
