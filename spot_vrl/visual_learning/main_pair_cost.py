import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data.dataset
import tqdm
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spot_vrl.visual_learning.datasets import (
    PairCostTrainingDataset,
    BaseTripletDataset,
    TripletHoldoutDataset,
)
from spot_vrl.imu_learning.losses import MarginRankingLoss
from spot_vrl.visual_learning.network import (
    CostNet,
    FullCostNet,
    EmbeddingNet,
)


class EmbeddingGenerator:
    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        train_set: BaseTripletDataset,
        holdout_set: BaseTripletDataset,
        tb_writer: SummaryWriter,
    ):
        self.batch_size = batch_size
        self.tb_writer = tb_writer

        self.train_tensors: Dict[str, torch.Tensor] = {}
        """Dict[terrain_type -> Tensor]"""

        self.holdout_tensors: Dict[str, torch.Tensor] = {}
        """Dict[terrain_type -> Tensor]"""

        for terrain, ds in sorted(train_set._categories.items()):
            self.train_tensors[terrain] = torch.cat(
                [ds[i][0][None, ...] for i in range(len(ds))], dim=0
            ).to(device)

        for terrain, ds in sorted(holdout_set._categories.items()):
            self.holdout_tensors[terrain] = torch.cat(
                [ds[i][0][None, ...] for i in range(len(ds))], dim=0
            ).to(device)

    @torch.no_grad()  # type: ignore
    def generate_plot(
        self, model: FullCostNet, dataset: Dict[str, torch.Tensor]
    ) -> plt.Figure:
        fig, ax = plt.subplots(sharey=True, constrained_layout=True)

        labels = []
        costs = []
        for label, tensors in dataset.items():
            """Perform batched computations for embeddings.

            The convolutional kernels allocate an extremely large amount of
            GPU memory if we pass in the entire dataset tensor at once.
            """
            embeddings = []
            for start in range(0, len(tensors), self.batch_size):
                """Perform batched computations for embeddings.

                The convolutional kernels allocate an extremely large amount of
                GPU memory if we pass in the entire dataset tensor at once.
                """
                end = min(start + self.batch_size, len(tensors))
                embeddings.append(model._encoder(tensors[start:end]))
            costs.append(model._cost_net(torch.cat(embeddings)).squeeze().cpu().numpy())
            labels.append(label)

        positions = np.arange(0, len(labels))

        ax.boxplot(costs, positions=positions, vert=False, showfliers=True)
        ax.set_yticks(positions, labels)
        ax.set_xlabel("Cost")

        return fig

    def write(self, model: FullCostNet, epoch: int) -> None:
        model.eval()
        with torch.no_grad():  # type: ignore
            self.tb_writer.add_figure(
                "train/costs",
                self.generate_plot(model, self.train_tensors),
                epoch,
                close=True,
            )  # type: ignore

            self.tb_writer.add_figure(
                "holdout/costs",
                self.generate_plot(model, self.holdout_tensors),
                epoch,
                close=True,
            )  # type: ignore


def train_epoch(
    train_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    model: FullCostNet,
    loss_fn: MarginRankingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    t1: Tensor
    t2: Tensor
    label: Tensor
    for t1, t2, label in train_loader:
        t1 = t1.to(device)
        t2 = t2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        cost1: Tensor
        cost2: Tensor
        cost1, cost2 = model(t1), model(t2)

        loss = loss_fn(cost1.squeeze(dim=1), cost2.squeeze(dim=1), label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def test_epoch(
    val_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    model: FullCostNet,
    loss_fn: MarginRankingLoss,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():  # type: ignore
        t1: Tensor
        t2: Tensor
        label: Tensor
        for t1, t2, label in val_loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            label = label.to(device)

            cost1: Tensor
            cost2: Tensor
            cost1, cost2 = model(t1), model(t2)

            loss = loss_fn(cost1.squeeze(dim=1), cost2.squeeze(dim=1), label)
            losses.append(loss.item())

    return sum(losses) / len(losses)


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--comment", type=str, default="")

    args = parser.parse_args()
    embedding_dim: int = args.embedding_dim
    encoder_path: Path = args.encoder_model
    dataset_dir: Path = args.dataset_dir
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

    # Set up the network and training parameters
    embedding_net = EmbeddingNet(embedding_dim)
    embedding_net.load_state_dict(
        torch.load(encoder_path, map_location=device),  # type: ignore
        strict=True,
    )
    embedding_net.requires_grad_(False)
    cost_net = CostNet(embedding_dim)

    model = FullCostNet(embedding_net, cost_net)
    model = model.to(device)

    loss_fn = MarginRankingLoss(margin)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

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

    logger.info("Loading EmbeddingGenerator data")
    embedder = EmbeddingGenerator(
        device,
        batch_size,
        cost_dataset.triplet_dataset,
        TripletHoldoutDataset().init_from_json(holdout_dataset_spec),
        tb_writer,
    )

    # Train the model
    pbar = tqdm.tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        torch.jit.script(model).save(save_dir / f"fullcostnet_{epoch:02d}.pt")
        val_loss = test_epoch(test_loader, model, loss_fn, device)

        tb_writer.add_scalar("train/loss", train_loss, epoch)  # type: ignore
        tb_writer.add_scalar("valid/loss", val_loss, epoch)  # type: ignore

        if embedder is not None:
            embedder.write(model, epoch)

        pbar.clear()


if __name__ == "__main__":
    main()
