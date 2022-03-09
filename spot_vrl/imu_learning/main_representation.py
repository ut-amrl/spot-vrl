import argparse
import os

import torch
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataset
from torch.optim import lr_scheduler

from spot_vrl.imu_learning.datasets import ManualTripletDataset
from spot_vrl.imu_learning.losses import TripletLoss
from spot_vrl.imu_learning.network import EmbeddingNet, TripletNet
from spot_vrl.imu_learning.trainer import fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)

    args = parser.parse_args()

    cuda = torch.cuda.is_available()

    # Set up data loaders
    triplet_dataset = ManualTripletDataset()
    batch_size = 1
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

    print("LOADED DATA")

    # Set up the network and training parameters
    margin = 48.0
    embedding_net = EmbeddingNet(triplet_dataset[0][0].shape, args.embedding_dim)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3 * batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)
    n_epochs = 80
    log_interval = 5

    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("GONNA TRAIN")
    fit(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        cuda,
        log_interval,
        args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
