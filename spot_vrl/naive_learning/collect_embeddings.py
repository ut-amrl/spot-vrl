# type: ignore

import torch

import os

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import argparse

from spot_vrl.naive_learning.network import EmbeddingNet, TripletNet
from spot_vrl.naive_learning.trainer import fit
import numpy as np

cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import random
import pickle

from spot_vrl.naive_learning.datasets import ConcreteGrassTripletDataset

batch_size = 40

parser = argparse.ArgumentParser()

parser.add_argument("--root_dirs", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--save_dir", default="embeddings", type=str)
parser.add_argument("--epoch", type=str)
parser.add_argument("--embedding_dim", type=int, default=6)
parser.add_argument("--cluster_count", type=int, default=6)
parser.add_argument("--skip_cluster", action="store_true")
parser.add_argument("--extraction_indices", nargs="+", type=int, default=[])
args = parser.parse_args()


def show_img(img, cluster_idx, save=True, member_count=None, gt=False):
    npimg = img.numpy().astype(np.uint8)
    fig = plt.figure()
    if not gt:
        plt.title("Cluster {}".format(cluster_idx))
    else:
        plt.title("GT Cluster {}".format(cluster_idx))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    if save:
        suffix = "cluster_{0}.png".format(cluster_idx)
        plt.imsave(args.save_dir + "/" + suffix, np.transpose(npimg, (1, 2, 0)))


def visualize_cluster_images(patches, labels=None, indices=None, num=16, gt=False):
    # Visualize the image patches
    if labels is not None:
        print("Num labels: ", len(np.unique(labels)))
        for lab in np.unique(labels):
            ind = np.where(labels == lab)

            patch_list = torch.tensor(patches[ind])
            if indices:
                patch_indices = np.array(indices)
                patch_list = patch_list[patch_indices]
            elif patch_list.shape[0] > num:
                patch_indices = np.random.choice(
                    list(range(patch_list.shape[0])), replace=False, size=(num)
                )
                patch_list = patch_list[patch_indices]
            show_img(
                make_grid(patch_list, nrow=int(patch_list.shape[0] / 4)),
                lab,
                member_count=patch_list.shape[0],
                gt=gt,
            )
    else:
        patch_list = torch.tensor(patches)
        if indices:
            patch_indices = np.array(indices)
            print("patch indices", patch_indices)
            patch_list = patch_list[patch_indices]
        elif patch_list.shape[0] > num:
            patch_indices = np.random.choice(
                list(range(patch_list.shape[0])), replace=False, size=(num)
            )
            patch_list = patch_list[patch_indices]
        show_img(
            make_grid(patch_list, nrow=int(patch_list.shape[0] / 4)),
            "extracted",
            member_count=patch_list.shape[0],
            gt=gt,
        )

        # full_img = transforms.ToPILImage()(full_img[0, :, :, :])
        # full_img = full_img.convert(mode = "RGB")


def plot_embeddings(
    embeddings, patches, labels=None, extraction_indices=None, xlim=None, ylim=None
):
    fig = plt.figure(figsize=(10, 10))
    if labels is not None:
        u_labels = np.unique(labels)
        # cmap = plt.cm.get_cmap("Pastel1", len(u_labels))
        cmap = plt.cm.get_cmap("Set1", len(u_labels))
        for idx in range(len(u_labels)):
            lab = u_labels[idx]
            ind = np.where(labels == lab)
            plt.gca().scatter(
                embeddings[ind, 0],
                embeddings[ind, 1],
                alpha=0.5,
                color=cmap(idx),
                label=lab,
                picker=True,
                pickradius=0.1,
            )
    else:
        plt.gca().scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            alpha=0.5,
            color="#ff3322",
            picker=True,
            pickradius=0.1,
        )
        extracted_indices = np.array(extraction_indices).astype(np.uint8)
        print("extracted indices", extracted_indices)
        plt.gca().scatter(
            embeddings[extracted_indices, 0],
            embeddings[extracted_indices, 1],
            alpha=1.0,
            color="#3322ff",
            picker=True,
            pickradius=0.1,
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings.png")

    visualize_cluster_images(patches, labels, indices=extraction_indices, num=32)
    plt.show()


def extract_embeddings(dataloader, model, embedding_dim=128):
    with torch.no_grad():
        embeddings = np.zeros((3 * len(dataloader), batch_size, embedding_dim))
        patches = np.zeros((3 * len(dataloader), batch_size, 3, 80, 80))
        labels = np.zeros((3 * len(dataloader), batch_size))
        k = 0
        if model:
            model.eval()
        for data, _ in dataloader:
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)

            labels[k] = 1
            labels[k + 1] = 1
            labels[k + 2] = 0
            for d in data:
                if model:
                    emb = model.get_embedding(d).data.cpu().numpy()
                    embeddings[k] = emb
                patches[k] = d.cpu().numpy()
                k += 1
        embeddings = embeddings[:k, :, :]
        patches = patches[:k, :, :, :]
        labels = labels[:k, :]
    return (
        embeddings.reshape(k * batch_size, embedding_dim),
        labels.reshape(k * batch_size),
        patches.reshape(k * batch_size, 3, 80, 80),
    )


print("Loading Model...")
model = None
if args.model_dir:
    embedding_net = EmbeddingNet(args.embedding_dim)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()

    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "trained_epoch_{}.pth".format(args.epoch)),
            map_location=torch.device("cpu"),
        )
    )

print("Constructing datasets...")
datasets = []
# for dir in args.root_dirs:
for traj_id in os.listdir(f"{args.root_dirs}/concrete"):
    datasets.append(ConcreteGrassTripletDataset(args.root_dirs, traj_id))

print("Building dataloader...")
triplet_test_dataset = torch.utils.data.ConcatDataset(datasets)
kwargs = {"num_workers": 8, "pin_memory": True} if cuda else {}
triplet_test_loader = torch.utils.data.DataLoader(
    triplet_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
)

print("Extracting embeddings...")
val_embeddings_tl, val_labels_tl, val_patches_tl = extract_embeddings(
    triplet_test_loader, model, args.embedding_dim
)

os.makedirs(args.save_dir, exist_ok=True)

visualize_cluster_images(val_patches_tl, val_labels_tl, num=32, gt=True)

if args.model_dir:
    if args.embedding_dim > 2:
        print("Running clustering...")
        clustering = KMeans(args.cluster_count).fit(val_embeddings_tl)
        labels = clustering.labels_
        print("Downprojecting for visualization...")
        embedded = TSNE().fit_transform(val_embeddings_tl)
        with open(
            os.path.join(
                args.model_dir, "clustering_{}.pkl".format(args.cluster_count)
            ),
            "wb",
        ) as f:
            pickle.dump(clustering, f)
    else:
        embedded = val_embeddings_tl
        # clustering = DBSCAN(eps=0.05, min_samples=10).fit(val_embeddings_tl).labels_
        labels = val_labels_tl
    print("Visualizing...")
    plot_embeddings(
        embedded,
        val_patches_tl,
        labels if not args.skip_cluster else None,
        args.extraction_indices,
    )
else:
    plt.show()
