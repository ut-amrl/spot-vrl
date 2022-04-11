import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tqdm
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from spot_vrl.imu_learning.datasets import BaseTripletDataset
from spot_vrl.imu_learning.losses import MarginRankingLoss
from spot_vrl.imu_learning.network import FullPairCostNet


class EmbeddingGenerator:
    def __init__(
        self,
        device: torch.device,
        train_set: BaseTripletDataset,
        holdout_set: BaseTripletDataset,
        tb_writer: SummaryWriter,
    ) -> None:
        self.tb_writer = tb_writer

        self.tensors: Dict[str, torch.Tensor] = {}
        """Dict[DatasetType -> Tensor]"""

        self.labels: Dict[str, List[str]] = {}
        """Dict[DatasetType -> List[str]]"""

        for terrain, ds in train_set._categories.items():
            t = self.tensors.get("train", torch.empty(0))
            t2 = torch.cat([ds[i][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["train"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("train", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for terrain, ds in holdout_set._categories.items():
            t = self.tensors.get("holdout", torch.empty(0))
            t2 = torch.cat([ds[i][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["holdout"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("holdout", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for key, val in self.tensors.items():
            self.tensors[key] = val.to(device)

    def write(self, model: FullPairCostNet, epoch: int) -> None:
        model.eval()
        with torch.no_grad():  # type: ignore
            for dataset_type, tensor in self.tensors.items():
                embeddings: Tensor = model.embedding_net(tensor)
                costs: Tensor = model.cost_net(embeddings)

                # SummaryWriter needs at least length 3 Tensors
                self.tb_writer.add_embedding(
                    costs.expand(-1, 3),
                    metadata=self.labels[dataset_type],
                    tag=f"embed-{dataset_type}",
                    global_step=epoch,
                )  # type: ignore


def fit(
    train_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    val_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    model: FullPairCostNet,
    loss_fn: MarginRankingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    n_epochs: int,
    device: torch.device,
    save_dir: Path,
    tb_writer: SummaryWriter,
    embedder: Optional[EmbeddingGenerator] = None,
) -> None:
    pbar = tqdm.tqdm(range(n_epochs), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
        )
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"trained_epoch_{epoch}.pth")
        )
        message = (
            f"Epoch {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}"
        )

        val_loss = test_epoch(val_loader, model, loss_fn, device)
        message += f"\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}"

        tb_writer.add_scalar("train/loss", train_loss, epoch)  # type: ignore
        tb_writer.add_scalar("valid/loss", val_loss, epoch)  # type: ignore

        if embedder is not None:
            embedder.write(model, epoch)

        pbar.clear()
        print(message)
        scheduler.step()


def train_epoch(
    train_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    model: FullPairCostNet,
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
        cost1, cost2 = model(t1, t2)

        loss = loss_fn(cost1, cost2, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def test_epoch(
    val_loader: DataLoader[Tuple[Tensor, Tensor, float]],
    model: FullPairCostNet,
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
            cost1, cost2 = model(t1, t2)

            loss = loss_fn(cost1, cost2, label)
            losses.append(loss.item())

    return sum(losses) / len(losses)
