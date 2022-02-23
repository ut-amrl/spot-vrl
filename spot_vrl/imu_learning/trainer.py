from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
import time


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    save_dir="ckpt",
    metrics=[],
    start_epoch=0,
    loss_input=False,
):
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
    )

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
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        for metric in metrics:
            message += "\t{}: {}".format(metric.name(), metric.value())

        print(message)
        scheduler.step()


def train_epoch(
    train_loader,
    model,
    loss_fn,
    optimizer,
    cuda,
    log_interval,
    metrics,
    loss_input=False,
):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

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


def test_epoch(val_loader, model, loss_fn, cuda, metrics, loss_input=False):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
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
