import argparse
import os
import time
from pathlib import Path
from spot_vrl.imu_learning.datasets import ManualTripletDataset, ManualTripletHoldoutSet

import torch
from loguru import logger
from spot_vrl.imu_learning.network import EmbeddingNet, TripletNet
from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    embedding_dim: int = args.embedding_dim
    cuda: bool = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    training_tensors = {}
    training_labels = []
    holdout_tensors = {}
    holdout_labels = []

    train_ds = ManualTripletDataset()
    for key, ds in train_ds._categories.items():
        training_tensors[key] = torch.cat(
            [ds[i][None, :] for i in range(len(ds))], dim=0
        )
        training_labels.extend([key for _ in range(len(ds))])
        if cuda:
            training_tensors[key] = training_tensors[key].cuda()

    # holdout_ds = ManualTripletHoldoutSet()
    holdout_ds = train_ds.holdout
    for key, ds in holdout_ds._categories.items():
        holdout_tensors[key] = torch.cat(
            [ds[i][None, :] for i in range(len(ds))], dim=0
        )
        holdout_labels.extend([key for _ in range(len(ds))])
        if cuda:
            holdout_tensors[key] = holdout_tensors[key].cuda()

    embedding_net = EmbeddingNet(train_ds[0][0].shape, embedding_dim)
    model = TripletNet(embedding_net)

    stime = time.strftime("%H-%M-%S")
    writer = SummaryWriter(log_dir=str(ckpt_dir / f"tensorboard-load-embeddings-{stime}"))  # type: ignore

    for epoch in range(40):
        epoch_file = ckpt_dir / f"trained_epoch_{epoch}.pth"
        if not os.path.exists(epoch_file):
            break

        logger.info(f"Loading state dict for epoch {epoch}")

        model.load_state_dict(torch.load(epoch_file))  # type: ignore
        if cuda:
            model.cuda()
        model.eval()
        with torch.no_grad():  # type: ignore
            logger.info(f"Generating embeddings for epoch {epoch}")
            training_embeddings = []
            holdout_embeddings = []

            for key, tensor in training_tensors.items():
                training_embeddings.append(model.get_embedding(tensor))

            for key, tensor in holdout_tensors.items():
                holdout_embeddings.append(model.get_embedding(tensor))

            writer.add_embedding(
                torch.cat(training_embeddings, dim=0),
                metadata=training_labels,
                tag="imu-train-embed",
                global_step=epoch,
            )  # type: ignore

            writer.add_embedding(
                torch.cat(holdout_embeddings, dim=0),
                metadata=holdout_labels,
                tag="imu-holdout-embed",
                global_step=epoch,
            )  # type: ignore


if __name__ == "__main__":
    main()
