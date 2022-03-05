import os
import time
from typing import Any, List, Tuple

import numpy as np
import torch
from spot_vrl.imu_learning.datasets import ManualTripletDataset, Triplet
from spot_vrl.imu_learning.losses import TripletLoss
from spot_vrl.imu_learning.network import TripletNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def fit(
    train_loader: DataLoader[Triplet],
    val_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: TripletLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    n_epochs: int,
    cuda: bool,
    log_interval: int,
    save_dir: str = "ckpt",
    metrics: List[Any] = [],
    start_epoch: int = 0,
    loss_input: bool = False,
) -> None:
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    stime = time.strftime("%H-%M-%S")
    slayers = "-".join([str(x) for x in model._embedding_net.sizes])
    writer = SummaryWriter(
        log_dir=os.path.join(save_dir, f"tensorboard-{slayers}--{stime}")
    )  # type: ignore

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            cuda,
            log_interval,
            metrics,
            loss_input,
        )
        # torch.save(
        #     model.state_dict(),
        #     os.path.join(save_dir, "trained_epoch_{}.pth".format(epoch)),
        # )
        message = "Epoch: {}/{}. Train set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, train_loss
        )
        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(
            val_loader, model, loss_fn, cuda, metrics, loss_input
        )
        val_loss /= len(val_loader)

        message += "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}".format(
            epoch + 1, n_epochs, val_loss
        )
        writer.add_scalar("train_loss", train_loss, epoch)  # type: ignore
        writer.add_scalar("val_loss", val_loss, epoch)  # type: ignore

        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        print(message)
        scheduler.step()

    m_ds = ManualTripletDataset()
    tensors = {}

    for key, ds in m_ds._categories.items():
        tensors[key] = torch.cat([ds[i][None, :] for i in range(len(ds))], dim=0)
        if cuda:
            tensors[key] = tensors[key].cuda()

    embeddings = []
    labels = []

    for key, t in tensors.items():
        embeddings.append(model.get_embedding(t))
        labels.extend([key for _ in range(t.shape[0])])

    writer.add_embedding(
        torch.cat(embeddings, dim=0),
        metadata=labels,
        tag="imu-embed",
        global_step=epoch,
    )  # type: ignore


def train_epoch(
    train_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cuda: bool,
    log_interval: int,
    metrics: List[Any],
    loss_input: bool = False,
) -> Tuple[float, List[Any]]:
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        target = None  # TODO(eyang): remove
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(data)

        if type(outputs) not in (tuple, list):
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze()
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
            if loss_input:
                loss_inputs += data

        loss_outputs = loss_fn(loss_inputs)
        loss = (
            torch.sum(torch.stack(loss_outputs))
            if type(loss_outputs) in (tuple, list)
            else loss_outputs
        )
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(losses),
            )
            for metric in metrics:
                message += "\t{}: {}".format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= batch_idx + 1
    return total_loss, metrics


def test_epoch(
    val_loader: DataLoader[Triplet],
    model: TripletNet,
    loss_fn: torch.nn.Module,
    cuda: bool,
    metrics: List[Any],
    loss_input: bool = False,
) -> Tuple[float, List[Any]]:
    with torch.no_grad():  # type: ignore
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0.0
        for batch_idx, data in enumerate(val_loader):
            target = None  # TODO(eyang): remove
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(data)

            if type(outputs) not in (tuple, list):
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze()
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
                if loss_input:
                    loss_inputs += data
            loss_outputs = loss_fn(loss_inputs)
            loss = (
                torch.sum(torch.stack(loss_outputs))
                if type(loss_outputs) in (tuple, list)
                else loss_outputs
            )
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
