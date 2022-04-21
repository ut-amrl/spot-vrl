import os
from pathlib import Path
from typing import Dict, List, Optional

import tqdm
import torch
from spot_vrl.visual_learning.datasets import (
    BaseTripletDataset,
    Triplet,
)
from spot_vrl.visual_learning.network import TripletNet
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class EmbeddingGenerator:
    def __init__(
        self,
        device: torch.device,
        train_set: BaseTripletDataset,
        holdout_set: BaseTripletDataset,
        tb_writer: SummaryWriter,
    ):
        self.tb_writer = tb_writer

        self.tensors: Dict[str, torch.Tensor] = {}
        """Dict[DatasetType -> Tensor]"""

        self.labels: Dict[str, List[str]] = {}
        """Dict[DatasetType -> List[str]]"""

        for terrain, ds in train_set._categories.items():
            t = self.tensors.get("train", torch.empty(0))
            t2 = torch.cat([ds[i][0][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["train"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("train", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for terrain, ds in holdout_set._categories.items():
            t = self.tensors.get("holdout", torch.empty(0))
            t2 = torch.cat([ds[i][0][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["holdout"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("holdout", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for key, val in self.tensors.items():
            self.tensors[key] = val.detach().to(device)

    def write(self, model: TripletNet, epoch: int) -> None:
        with torch.no_grad():  # type: ignore
            model.eval()
            for dataset_type, tensor in self.tensors.items():
                self.tb_writer.add_embedding(
                    model.get_embedding(tensor),
                    metadata=self.labels[dataset_type],
                    tag=f"embed-{dataset_type}",
                    global_step=epoch,
                )  # type: ignore


def fit(
    train_loader: DataLoader[Triplet],
    val_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: torch.nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    n_epochs: int,
    device: torch.device,
    save_dir: Path,
    tb_writer: SummaryWriter,
    embedder: Optional[EmbeddingGenerator] = None,
) -> None:
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
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
            model.state_dict(),
            os.path.join(save_dir, "trained_epoch_{}.pth".format(epoch)),
        )

        val_loss = test_epoch(val_loader, model, loss_fn, device)

        tb_writer.add_scalar("train_loss", train_loss, epoch)  # type: ignore
        tb_writer.add_scalar("val_loss", val_loss, epoch)  # type: ignore

        if embedder is not None and (epoch + 1) % 10 == 0:
            embedder.write(model, epoch)

        pbar.clear()
        scheduler.step()


def train_epoch(
    train_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: torch.nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    triplet: Triplet
    for triplet in train_loader:
        triplet = tuple(t.to(device) for t in triplet)  # type: ignore

        optimizer.zero_grad()
        embeddings: Triplet = model(triplet)

        loss: Tensor = loss_fn(*embeddings)
        losses.append(loss.item())
        loss.backward()  # type: ignore
        optimizer.step()

    return sum(losses) / len(losses)


def test_epoch(
    val_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: torch.nn.TripletMarginLoss,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():  # type: ignore
        triplet: Triplet
        for triplet in val_loader:
            triplet = tuple(t.to(device) for t in triplet)  # type: ignore

            embeddings: Triplet = model(triplet)
            loss: Tensor = loss_fn(*embeddings)
            losses.append(loss.item())

    return sum(losses) / len(losses)
