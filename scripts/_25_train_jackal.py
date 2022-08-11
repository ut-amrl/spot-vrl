"""Contains the first part of the model (creating representation, saving resulting k-means model and encoders)"""
__author__= "Daniel Farkash, Haresh Karnan"
__email__= "dmf248@cornell.edu"
__date__= "August 10, 2022"


from genericpath import exists
import glob
import pytorch_lightning as pl
import torch
from torch import uint8
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from scipy import fftpack, gradient
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from termcolor import cprint
import cv2
import random
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import List, Union, Tuple
import os
import yaml
from lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import cluster_jackal
from dict_custom_data import CustomDataset, MyDataLoader
from PIL import Image

def exclude_bias_and_norm(p):
    return p.ndim == 1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 2, 2)

class RepModel(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=64, inertial_shape=None,
                 scale_loss:float=1.0/32, lambd:float=3.9e-6, weight_decay=1e-6,
                 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1
                 , l1_coeff = 1, imu_in_rep=True ,save = False):
        super(RepModel, self).__init__()

        self.save_hyperparameters(
            'lr',
            'latent_size',
            'inertial_shape',
            'scale_loss',
            'lambd',
            'weight_decay',
            'l1_coeff'
        )

        self.scale_loss = scale_loss
        self.lambd = lambd
        self.weight_decay = weight_decay
        self.lr = lr
        self.per_device_batch_size = per_device_batch_size
        self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio
        self.l1_coeff = l1_coeff
        self.save = save
        self.imu_in_rep = imu_in_rep

        # visual encoder architecture
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False), # 31 x 31
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), # 15 x 15
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False), # 7 x 7
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False), # 3 x 3 
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Flatten(), # 256 output
            nn.Linear(128, 128)
        )

        # inertial encoder architecture
        self.inertial_encoder = nn.Sequential(
            nn.Linear(600+6, 256, bias=False), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 128)
        )

        # projector archiecture
        self.projector = nn.Sequential(
            nn.Linear(128, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False)
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)
  
        # coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, main_patch_lst, inertial_data, patch_list_1, patch_list_2 ):
        visual_patch = main_patch_lst[0]
        imu_history = inertial_data

        # Encode and project main patch and corresponding inertial data (used for Lss 1)
        v_encoded = self.visual_encoder(visual_patch)
        i_encoded = self.inertial_encoder(imu_history)
        z1 = self.projector(v_encoded)
        z2 = self.projector(i_encoded)

        lst1_encoded = []
        lst2_encoded = []
        lst1_projected =[]
        lst2_projected =[]

        # Encode and project all grid patches are used for Lss 2
        for i in range(25):
            lst1_encoded.append(self.visual_encoder(patch_list_1[i])) 
            lst2_encoded.append(self.visual_encoder(patch_list_2[i])) 

        for i in range(25):
            lst1_projected.append(self.projector(lst1_encoded[i])) 
            lst2_projected.append(self.projector(lst2_encoded[i]))
  
        return z1, z2, lst1_projected, lst2_projected

    # Compute Variance Invariance Covariance Regularization loss
    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
  
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2

        cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + self.off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])
  
        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def all_reduce(self, c):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

    def training_step(self, batch, batch_idx):

        # ensure that models are trainable
        self.visual_encoder.train()
        self.inertial_encoder.train()
        self.projector.train()

        main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch

        # forward pass
        visual_emb, inertial_emb, lst1_emb, lst2_emb = self(main_patch_lst, inertial_data, patch_list_1, patch_list_2)

        # compute Lss 1
        lss1 = self.vicreg_loss(visual_emb, inertial_emb)

        # compute Lss 2
        lss2_lst = torch.zeros(25)

        for i in range(25):
            v_emb_1 = lst1_emb[i]
            v_emb_2 = lst2_emb[i]
            lss2_lst[i] = self.vicreg_loss(v_emb_1, v_emb_2)

        lss2 = torch.mean(lss2_lst)

        # calculate combined loss
        loss = self.l1_coeff*lss1 + (1- self.l1_coeff)*lss2
        
        # save record of loss for monitoring
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        # ensure that models do not trai on validation
        self.visual_encoder.eval()
        self.inertial_encoder.eval()
        self.projector.eval()

        main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch

        # forward pass
        visual_emb, inertial_emb, lst1_emb, lst2_emb = self(main_patch_lst, inertial_data, patch_list_1, patch_list_2)

        # compute Lss 1
        lss1 = self.vicreg_loss(visual_emb, inertial_emb)

        # compute Lss 2
        lss2_lst = torch.zeros(25)

        for i in range(25):
            v_emb_1 = lst1_emb[i]
            v_emb_2 = lst2_emb[i]
            lss2_lst[i] = self.vicreg_loss(v_emb_1, v_emb_2)

        lss2 = torch.mean(lss2_lst)

        # calculate combined loss
        loss = self.l1_coeff*lss1 + (1- self.l1_coeff)*lss2

        # save record of Lss 1, and Lss 2, and combined losses for monitoring
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log('val_loss_l1', lss1, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        self.log('val_loss_l2', lss2, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):

        # Use AdamW optimizer and no learning rate scheduling 
        # Used linear warmup cosine annealing in the past, but it is not necessary
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

        return optimizer

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):

        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 2 == 0 or self.current_epoch == self.trainer.max_epochs-1:

            main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch

            # formatting
            visual_patch = main_patch_lst[0]
            label = np.asarray(label) 
            visual_patch = visual_patch.float()
            inertial_data = inertial_data.float()

            # get encodings
            with torch.no_grad():
                visual_encoding = self.visual_encoder(visual_patch.cuda())
                inertial_encoding = self.inertial_encoder(inertial_data.cuda())

            visual_encoding = visual_encoding.cpu()
            inertial_encoding = inertial_encoding.cpu()
            visual_patch = visual_patch.cpu()
            
            # create lists for fist batch and append to them for following batches
            if batch_idx == 0:
                self.visual_encoding = [visual_encoding[:, :]]
                self.inertial_encoding = [inertial_encoding[:, :]]
                self.visual_patch = [visual_patch[:, :, :, :]]
                self.label = label[:]
            else:
                self.visual_patch.append(visual_patch[:, :, :, :])
                self.visual_encoding.append(visual_encoding[:, :])
                self.inertial_encoding.append(inertial_encoding[:, :])
                self.label = np.concatenate((self.label, label[:]))

    # Find random groups of 25 images from each cluster
    def sample_clusters(self,clusters,elbow,vis_patch):

            # initialize
            dic = {}
            for a in range(elbow):
                dic[a] = []

            # For each cluster, find indexes of images in that cluster and extract 25 of them
            for i in range(elbow):

                idx = np.where(clusters == i)

                for j in range(25):

                    # select correct patch
                    chosen = np.random.randint(low=0,high=len(idx[0]))
                    vp = vis_patch[idx[0][chosen], :, :, :]

                    # formatting for displayable image
                    vp = vp.cpu()
                    vp = vp.numpy()
                    vp= (vp * 255).astype(np.uint8)
                    vp = np.moveaxis(vp, 0, -1)

                    dic[i].append(vp)

            return dic, elbow

    # create and save 25 image grids for each cluster from dictionary image info
    # TODO: change file that images are saved to
    def img_clusters(self,dic, elbow):

        for i in range(elbow):

            # initialize grid
            new_im = Image.new('RGB', (64*5,64*5))

            for j in range(25):

                vp = dic[i][j]
    
                # patch number to grid location
                h = int(j/5)
                w = j%5

                # format and paste individual patches to grid
                im = Image.fromarray(vp)
                im = im.convert('RGB')
                im.thumbnail((64,64))
                new_im.paste(im, (h*64,w*64))

            # save grid image
            new_im.save("/home/dfarkash/garbage" +"/group"+str(i)+".png")

    def on_validation_end(self) -> None:
        
        # log visualization data only every other epoch or during the last epoch
        if self.current_epoch % 2 == 0 or self.current_epoch == self.trainer.max_epochs-1:

            # catenate patch/encoding lists
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)

            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)

            # limit number of patches used to not fill memory
            ve = self.visual_encoding[idx[:2000],:]
            ie = self.inertial_encoding[idx[:2000],:]
            vis_patch = self.visual_patch[idx[:2000],:,:,:]

            # deal with whether or not the inertial data should be included in the representation
            if self.imu_in_rep:
                data = torch.cat((ve, ie), dim=1)
            else:
                data = ve

            # get results of k-means model clustering
            clusters, elbow, model = cluster_jackal.cluster_model(data)

            # Save the cluster image grids on the final epoch only
            if self.current_epoch == self.trainer.max_epochs-1: 
                a,b = self.sample_clusters(clusters,elbow, vis_patch)
                self.img_clusters(a,b)

            # Save the k-means, visual encoder, and inertial encoder models on the last epoch for only one of the gpus used
            # only if user specified that they should be saved (uses all data instead of only 2000)
            # TODO: change folder to which the models are saved
            if self.current_epoch == self.trainer.max_epochs-1 and self.save and torch.cuda.current_device() == 0:

                v = self.visual_encoding[idx,:]
                i = self.inertial_encoding[idx,:]

                if self.imu_in_rep:
                    d = torch.cat((v, i), dim=1)
                else:
                    d = v

                print("  Saved model: ")
                # calculate and print accuracy
                o= cluster_jackal.accuracy_naive_model(d,self.label[idx])
                
                # save k-means model
                with open("/home/dfarkash/no_inert_rep_cost_data/model.pkl", "wb") as f:
                    pickle.dump(o, f)

                # save the visual encoder
                torch.save(self.visual_encoder.state_dict(), "/home/dfarkash/no_inert_rep_cost_data/visual_encoder.pt")
                # save the inertial encoder
                torch.save(self.inertial_encoder.state_dict(), "/home/dfarkash/no_inert_rep_cost_data/inertial_encoder.pt")
                
            # create labels/grouping for visualization
            metadata = list(zip(self.label[idx[:2000]], clusters))
            metadata_header = ["labels","clusters"]

            # calculate and print accuracy
            out = cluster_jackal.accuracy_naive(data,self.label[idx[:2000]])

            # log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar("K-means accuracy", out, self.current_epoch)

            self.logger.experiment.add_embedding(mat=data,
                                                 label_img=self.visual_patch[idx[:2000], :, :, :],
                                                 global_step=self.current_epoch,
                                                 metadata=metadata,
                                                 metadata_header = metadata_header
                                               )

            del self.visual_patch, self.visual_encoding, self.label


    @property
    def total_training_steps(self) -> int:
        dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = self.trainer.tpu_cores if self.trainer.tpu_cores else self.trainer.num_processes
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> int:
        return num_warmup_steps * num_training_steps if isinstance(num_warmup_steps, float) else num_training_steps

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Dual Auto Encoder')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--l1_coeff', type=float, default=1, metavar='L1C',
                        help='L1 loss coefficient (1)')
    parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
                        help='log directory (default: logs)')
    parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
                        help='model directory (default: models)')
    parser.add_argument('--num_gpus', type=int, default=2, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='Size of the common latent space (default: 512)')
    parser.add_argument('--save', type=int, default=0, metavar='N',
                        help='Whether to save the k means model and encoders at the end of the run')
    parser.add_argument('--imu_in_rep', type=int, default=1, metavar='N',
                        help='Whether to include the inertial data in the representation')
    parser.add_argument('--dataset_config_path', type=str, default='jackal_data/dataset_config_haresh_local.yaml')
    args = parser.parse_args()
    
    # check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloader and model using arguments
    dm = MyDataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size)

    model = RepModel(lr=args.lr, latent_size=args.latent_size,
                        inertial_shape=1200, scale_loss=1.0, lambd=1./args.latent_size, 
                          per_device_batch_size=args.batch_size, l1_coeff = args.l1_coeff, 
                          imu_in_rep = bool(args.imu_in_rep), save = bool(args.save)).to(device)

    early_stopping_cb = EarlyStopping(monitor='train_loss', mode='min', min_delta=0.00, patience=1000)
    
    # create model checkpoint only at end of training
    model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
                                          filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_', verbose=True)

    print("Training model...")
    trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
                         max_epochs=args.epochs,
                         callbacks=[model_checkpoint_cb],
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         logger=True,
                         )

    # fit the model
    trainer.fit(model, dm)