"""Contains the second model (using user preferences, saved items from part 1 to learn cost network)"""
__author__= "Daniel Farkash"
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
from s2_custom_data import s2CustomDataset, s2DataLoader
from PIL import Image
from _25_train_jackal import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
import jenkspy

def exclude_bias_and_norm(p):
    return p.ndim == 1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 2, 2)

class s2Model(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=256, inertial_shape=None,
                 scale_loss:float=1.0/32, lambd:float=3.9e-6, weight_decay=1e-6,
                 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1, 
                 full = True, model_folder = "cost_data"):
        super(s2Model, self).__init__()

        self.save_hyperparameters(
            'lr',
            'latent_size',
            'inertial_shape',
            'scale_loss',
            'lambd',
            'weight_decay',
        )

        self.scale_loss = scale_loss
        self.lambd = lambd
        self.weight_decay = weight_decay
        self.lr = lr
        self.per_device_batch_size = per_device_batch_size
        self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio
        self.full = full
        self.model_folder = model_folder

        # Load saved visual encoder from model foder
        visual_encoder = RepModel().visual_encoder
        visual_encoder.load_state_dict(torch.load("/home/dfarkash/"+self.model_folder+"/visual_encoder.pt"))
        visual_encoder.eval()
        self.visual_encoder = visual_encoder
        
        # Ensure that model will not be updated during training
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # If both the visual and inertial models are used ("full")
        # then load the saved inertial incoder form the model folder
        # and ensure that it will not be updated during training
        if self.full:
            inertial_encoder = RepModel().inertial_encoder
            inertial_encoder.load_state_dict(torch.load("/home/dfarkash/"+self.model_folder+"/inertial_encoder.pt"))
            inertial_encoder.eval()
            self.inertial_encoder = inertial_encoder

            for param in self.inertial_encoder.parameters():
                param.requires_grad = False

        # Log user specified preferences
        # TODO: change path to point to desired user preferences 
        # (array of ints) corresponding to ranking of terrains created
        # by GUI (lower value corresponds to higher priority)
        preferences_path = "/home/dfarkash/cost_data/preferences.pkl"
        cprint('Loading user preferences from {}'.format(preferences_path))
        self.preferences = pickle.load(open(preferences_path, 'rb'))

        # quick override fors testing
        self.preferences = [0,6,1,2,5,4,3]

        # load k-means model from model folder
        model_path = "/home/dfarkash/"+self.model_folder+"/model.pkl"
        cprint('Loading cluster_model from {}'.format(model_path))
        self.model = pickle.load(open(model_path, 'rb'))

        # simple architecture for cost network
        self.cost_net = nn.Sequential(
            nn.Linear(128, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, 1), nn.ReLU(inplace=True)
        )

    # Input patches go through saved visual encoder then cost net to get cost
    def forward(self, patch):

        with torch.no_grad():
            v_encoded = self.visual_encoder(patch)

        cost = self.cost_net(v_encoded)
        
        return cost

    def find_true_cost(self, patch, inertial):

        # deal with inclusion of inertial data
        self.visual_encoder.eval()
        if self.full:
            self.inertial_encoder.eval()
        
        with torch.no_grad():
            visual_encoding = self.visual_encoder(patch)
            if self.full:
                inertial_encoding = self.inertial_encoder(inertial)

        if self.full:
            representation = torch.cat((visual_encoding, inertial_encoding), dim=1)
        else:
            representation = visual_encoding

        # unscaled representation (used scaled in past)
        representation=representation.cpu()
        representation=representation.detach().numpy()

        # use k-means model to classify representations
        cluster = self.model.predict(representation)
        cluster = cluster.astype('int')

        # use user preferences tofind 'true cost" (ranking)
        true_cost = []
        
        for i in range(len(cluster)):
            true_cost.append(self.preferences[cluster[i]])

        true_cost = torch.Tensor(true_cost).cuda()

        return true_cost


    # softmax calculation
    def smax(self, s1, s2):
        return (torch.exp(s1) / (torch.exp(s1) + torch.exp(s2))) 

    def training_step(self, batch, batch_idx):
        self.visual_encoder.eval()

        if self.full:
            self.inertial_encoder.eval()
        self.cost_net.train()

        patch, inertial = batch

        # find ranking/preference by classifying using saved k-means model + user preferences
        true_cost = self.find_true_cost(patch, inertial)

        # forward pass
        cost = self(patch)

        # Toggle between using simple linear costs and softmax loss
        linear = False

        if linear :

            # linear costs
            mse = torch.nn.MSELoss()
            loss = mse(cost[:,0], true_cost)

        else: 

            loss_lst = torch.zeros(len(true_cost)-1)
            i = 0

            # find pairs by iterating through batch and applying softmax 
            # loss to pairs with different priorities and using mse if they are the same
            while i < len(true_cost)-1:
                if true_cost[i] < true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i,0], cost[i+1,0])
                elif true_cost[i] > true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i+1,0], cost[i,0])
                else:
                    loss_lst[i] = (true_cost[i] - true_cost[i+1])**2
                
                # penalize loss from going above 100.0
                if cost[i,0] > 100.0:
                    loss_lst[i] += (cost[i, 0]- 100.0)**2
                i=i+1

            loss = torch.mean(loss_lst)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.visual_encoder.eval()
        
        if self.full:
            self.inertial_encoder.eval()
        self.cost_net.eval()

        patch, inertial = batch

        # find ranking/preference by classifying using saved k-means model + user preferences
        true_cost = self.find_true_cost(patch, inertial)

        # forward pass
        cost = self(patch)

        # Toggle between using simple linear costs and softmax loss
        linear = False

        if linear :

            # linear costs
            mse = torch.nn.MSELoss()
            loss = mse(cost[:,0], true_cost)

        else: 

            loss_lst = torch.zeros(len(true_cost)-1)
            i = 0

            # find pairs by iterating through batch and applying softmax 
            # loss to pairs with different priorities and using mse if they are the same
            while i < len(true_cost)-1:
                if true_cost[i] < true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i,0], cost[i+1,0])
                elif true_cost[i] > true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i+1,0], cost[i,0])
                else:
                    loss_lst[i] = (true_cost[i] - true_cost[i+1])**2
                
                # penalize loss from going above 100.0
                if cost[i,0] > 100.0:
                    loss_lst[i] += (cost[i, 0]- 100.0)**2
                i=i+1

            loss = torch.mean(loss_lst)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    # Use adamw optimizer with no learning rate scheduling
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        return optimizer
    
    # save batch data for later use in visualizations/data analysis
    # see _25_train_jackal for general understanding
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):

        if self.current_epoch % 2 == 0:

            patch, inertia = batch
            true_cost = self.find_true_cost(patch,inertia)
            true_cost = true_cost.cpu()
            true_cost = np.asarray(true_cost)
            patch = patch.float()
            
            with torch.no_grad():
                visual_encoding = self.visual_encoder(patch.cuda())
                cost = self.cost_net(visual_encoding)
            visual_encoding = visual_encoding.cpu()
            cost = cost.cpu()
            
            if batch_idx == 0:
                self.visual_encoding = [visual_encoding[:, :]]
                self.patch = [patch[:, :, :, :]]
                self.true_cost = true_cost[:]
                self.cost = cost[:]
            elif batch_idx % 1 == 0:
                self.patch.append(patch[:, :, :, :])
                self.visual_encoding.append(visual_encoding[:, :])
                self.true_cost = np.concatenate((self.true_cost, true_cost[:]))
                self.cost = np.concatenate((self.cost, cost[:]))

    # find the accuracy of the binned rankings compared to the true rankings
    # TODO: set ranges equal to the number of different rankings
    def accuracy_naive(self, binned_cost, true_cost):

        best_part_accs = []
        best_part_lens = []

        for i in range(7):
            best_part_acc = 0
            inds = np.where(binned_cost==i)[0]
            for j in range(7):
                part_acc = np.sum(true_cost[inds] == self.preferences[j])/len(inds)
                if part_acc > best_part_acc:
                    best_part_acc = part_acc
            
            best_part_accs.append(best_part_acc)
            best_part_lens.append(len(inds))
        
        best_acc = np.sum(np.array(best_part_accs)*np.array(best_part_lens))/np.sum(best_part_lens)

        print("accuracy:")
        print(best_acc)
        print("best_part_accs:")
        print(best_part_accs)
        print("best_part_lens:")
        print(best_part_lens)
        return best_acc

    # Create a scatter/box plot projection that shows the 
    # distribution of costs relative to the terrains
    def scatter(self, true_cost, cost, vis_patch):

        tc = true_cost
        c = cost

        # params for data binning
        dict = {}
        num_bins = 100
        min_size = 0
        max_size = 40
        bin_size = (max_size - min_size)/num_bins

        # bin data based on cost values
        for i in range(num_bins):
            dict[i] = []
        dict["sizes"] = []
        for j in range(len(tc)):
            bin = int(c[j]/bin_size)
            dict[bin].append((tc[j],vis_patch[j,:,:,:],c[j]))

        for k in range (num_bins):
            dict["sizes"].append(len(dict[k]))

        max_size = max(dict["sizes"])

        # initialize new data rep
        new_mat = torch.zeros((len(tc), 3))
        new_true_cost = torch.zeros(len(tc))
        new_cost = torch.zeros(len(tc))
        new_vis_patch = torch.zeros((len(tc),3,64,64))

        # create new embedding with binned cost values
        tot = 0
        for l in range(num_bins):
            for m in range(dict["sizes"][l]):
                new_mat[tot, 0] = l*5
                new_mat[tot, 1] = m
                new_mat[tot, 2] = 0
                new_true_cost[tot] = torch.from_numpy(np.asarray(dict[l][m][0])).to(new_true_cost)
                new_cost[tot] = torch.from_numpy(dict[l][m][2]).to(new_cost)
                new_vis_patch[tot, :,:,:]= (dict[l][m][1]).to(new_vis_patch)
                tot=tot+1

        # lables for new embedding
        metadata = list(zip(new_cost, new_true_cost))
        metadata_header = ["costs","true_costs"]

        # add the embedding to tensorboard
        self.logger.experiment.add_embedding(new_mat,
                    label_img=new_vis_patch,
                    global_step=self.current_epoch,
                    metadata=metadata,
                    metadata_header = metadata_header)
        

        
        


    def on_validation_end(self) -> None:
        
        # log visualization data only every other epoch or during the last epoch
        if self.current_epoch % 2 == 0 and torch.cuda.current_device() == 0:
            
            # catenate patch/encoding lists
            self.patch = torch.cat(self.patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            
            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)

            ve = self.visual_encoding[idx,:]
            vis_patch = self.patch[idx,:,:,:]
            cost = self.cost[idx]
            true_cost = self.true_cost[idx]

            # use pandas/jenkspy to binn the costs for comparison to user preferences
            # TODO: set nb_class to the number of distinct user rankings
            cost_frame = pd.DataFrame(cost[:,0])
            breaks = jenkspy.jenks_breaks(cost_frame[0] ,nb_class=7) 
            cost_bin = pd.cut(cost_frame[0], bins=breaks, labels=False, include_lowest=True)

            # calculate and log the accuracy of the independently binned costs (found by model)
            # compared to the true costs (user rankings)
            acc = self.accuracy_naive(cost_bin, true_cost)
            self.logger.experiment.add_scalar("Binned accuracy", acc, self.current_epoch)

            # log projection of encoded terrains for visualization of binned vs true clustering
            # TODO: set to True if this is desired, set to false if scatter/bar plot is desired
            if False:
                metadata = list(zip(cost[:2000], true_cost[:2000],cost_bin[:2000]))
                metadata_header = ["costs","true_costs","cost bins"]

                mat=ve[:2000]
                
                self.logger.experiment.add_embedding(mat,
                                                    label_img=vis_patch[:2000],
                                                    global_step=self.current_epoch,
                                                    metadata=metadata,
                                                    metadata_header=metadata_header)
            else:
                self.scatter(true_cost[:500],cost[:500], vis_patch[:500])
                                               

            del self.patch, self.visual_encoding, self.cost, self.true_cost



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
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
                        help='log directory (default: logs)')
    parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
                        help='model directory (default: models)')
    parser.add_argument('--num_gpus', type=int, default=4, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=256, metavar='N',
                        help='Size of the common latent space (default: 512)')
    parser.add_argument('--dataset_config_path', type=str, default='jackal_data/different.yaml')
    parser.add_argument('--model_folder', type=str, default='cost_data', help = "Folder in home/dfarkash that contains the s1 encoders and k-means model")
    parser.add_argument('--full', type=int, default=1, metavar='N',
                        help='Whether to use the whole model')
    args = parser.parse_args()
    
    # check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloader and model using arguments
    dm = s2DataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size,
                        full = bool(args.full))

    model = s2Model(lr=args.lr, latent_size=args.latent_size,
                        inertial_shape=1200, scale_loss=1.0, lambd=1./args.latent_size, 
                          per_device_batch_size=args.batch_size, full=bool(args.full), 
                          model_folder = args.model_folder).to(device)


    early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)

    # checkpoint new model whenever validation loss reaches a new minimum
    model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
                                          filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
                                          monitor='val_loss', verbose=True)

    print("Training model...")
    trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
                         max_epochs=args.epochs,
                         callbacks=[model_checkpoint_cb],
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         logger=True,
                         sync_batchnorm=True,
                         )

    # fit the model
    trainer.fit(model, dm)