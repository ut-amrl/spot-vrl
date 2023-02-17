import os
from pathlib import Path
from typing import Dict, List, Optional

import tqdm
import torch
from loguru import logger
from spot_vrl.visual_learning.datasets import (
    BaseTripletDataset,
    Triplet,
)
from spot_vrl.visual_learning.network import EmbeddingNet
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# Replace the unbuffered torch.utils.tensorboard._embedding.make_mat with
# this function for networked filesystems.
# def buffered_make_mat(matlist, save_path):
#     import io

#     with io.BytesIO() as buffer:
#         for x in matlist:
#             x = [str(i.item()) for i in x]
#             buffer.write(tf.compat.as_bytes("\t".join(x) + "\n"))

#         fs = tf.io.gfile.get_filesystem(save_path)
#         with tf.io.gfile.GFile(fs.join(save_path, "tensors.tsv"), "wb") as f:
#             f.write(buffer.getvalue())


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

        self.tensors: Dict[str, torch.Tensor] = {}
        """Dict[DatasetType -> Tensor]"""

        self.labels: Dict[str, List[str]] = {}
        """Dict[DatasetType -> List[str]]"""

        for terrain, ds in sorted(train_set._categories.items()):
            t = self.tensors.get("train", torch.empty(0))
            t2 = torch.cat([ds[i][1][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["train"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("train", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for terrain, ds in sorted(holdout_set._categories.items()):
            t = self.tensors.get("holdout", torch.empty(0))
            t2 = torch.cat([ds[i][1][None, ...] for i in range(len(ds))], dim=0)
            self.tensors["holdout"] = torch.cat((t, t2), dim=0)
            self.labels.setdefault("holdout", []).extend(
                [terrain for _ in range(len(ds))]
            )

        for key, val in self.tensors.items():
            self.tensors[key] = val.to(device)

    def write(self, model: EmbeddingNet, epoch: int) -> None:
        model.eval()
        with torch.no_grad():  # type: ignore
            for dataset_type, tensor in self.tensors.items():
                """Perform batched computations for embeddings.

                The convolutional kernels allocate an extremely large amount of
                GPU memory if we pass in the entire dataset tensor at once.
                """
                embeddings = []
                for start in range(0, tensor.size(dim=0), self.batch_size):
                    end = min(start + self.batch_size, tensor.size(dim=0))
                    embeddings.append(model(tensor[start:end]))

                self.tb_writer.add_embedding(
                    torch.cat(embeddings),
                    metadata=self.labels[dataset_type],
                    tag=f"embed-{dataset_type}",
                    global_step=epoch,
                )  # type: ignore


def fit(
    train_loader: DataLoader[Triplet],
    val_loader: DataLoader[Triplet],
    model: EmbeddingNet,
    loss_fn: torch.nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    start_epoch: int,
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
    pbar = tqdm.tqdm(range(start_epoch, start_epoch + n_epochs), desc="Training")
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

        tb_writer.add_scalar("train/loss", train_loss, epoch)  # type: ignore
        tb_writer.add_scalar("valid/loss", val_loss, epoch)  # type: ignore

        if embedder is not None and (epoch + 1) % 10 == 0:
            embedder.write(model, epoch)

        pbar.clear()
        scheduler.step(val_loss)

    # evaluation-only case
    if n_epochs == 0 and embedder is not None:
        embedder.write(model, start_epoch)


def train_epoch(
    train_loader: DataLoader[Triplet],
    model: EmbeddingNet,
    loss_fn: torch.nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    triplet: Triplet
    for triplet in train_loader:
        triplet = tuple(t.to(device) for t in triplet)  # type: ignore
        anchor, pos, neg = triplet

        optimizer.zero_grad()
        e_anchor = model(anchor)
        e_pos = model(pos)
        e_neg = model(neg)

        loss: Tensor = loss_fn(e_anchor, e_pos, e_neg)
        losses.append(loss.item())
        loss.backward()  # type: ignore
        optimizer.step()

    return sum(losses) / len(losses)


def test_epoch(
    val_loader: DataLoader[Triplet],
    model: EmbeddingNet,
    loss_fn: torch.nn.TripletMarginLoss,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    with torch.no_grad():  # type: ignore
        triplet: Triplet
        for triplet in val_loader:
            triplet = tuple(t.to(device) for t in triplet)  # type: ignore
            anchor, pos, neg = triplet

            e_anchor = model(anchor)
            e_pos = model(pos)
            e_neg = model(neg)
            loss: Tensor = loss_fn(e_anchor, e_pos, e_neg)
            losses.append(loss.item())

    return sum(losses) / len(losses)
