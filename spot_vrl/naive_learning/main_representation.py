import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, required=True)
# parser.add_argument('--trajectories', type=str, nargs='+', required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--embedding_dim', type=int, required=True)
parser.add_argument('--grayscale', action='store_true')

args = parser.parse_args()

from spot_vrl.naive_learning.trainer import fit
cuda = torch.cuda.is_available()

# Set up data loaders
from spot_vrl.naive_learning.datasets import ConcreteGrassTripletDataset
pref_datasets = []
for traj_id in os.listdir(f"{args.root_dir}/concrete"):
    pref_datasets.append(ConcreteGrassTripletDataset(args.root_dir, traj_id))
triplet_dataset = torch.utils.data.ConcatDataset(pref_datasets)
batch_size = 128
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
train_size = int(len(triplet_dataset) * 0.75)
train_set, test_set = torch.utils.data.dataset.random_split(triplet_dataset, (train_size, len(triplet_dataset) - train_size))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

print("LOADED DATA")

# Set up the network and training parameters
from spot_vrl.naive_learning.network import EmbeddingNet, TripletNet
from spot_vrl.naive_learning.losses import TripletLoss

margin = 2.
embedding_net = EmbeddingNet(args.embedding_dim)
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 40
log_interval = 5

if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)

print("GONNA TRAIN")
fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, args.ckpt_dir)
