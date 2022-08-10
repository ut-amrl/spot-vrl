from genericpath import exists
import glob
#from cluster_spot.scripts.train_jackal import MyDataLoader
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
# from tf.keras.optimizers.schedules import CosineDecay
# import librosa
# import librosa.display as display
from lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
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

        # net_path = "/home/dfarkash/spot-vrl/models/26-07-2022-11-06-47_.ckpt"
        # cprint('Loading model from {}'.format(net_path))
        # net = BarlowModel.load_from_checkpoint(net_path)
        # net.eval()
        # self.visual_encoder = net.visual_encoder

        # print(self.full)

        visual_encoder = BarlowModel().visual_encoder
        visual_encoder.load_state_dict(torch.load("/home/dfarkash/"+self.model_folder+"/visual_encoder.pt"))
        visual_encoder.eval()
        self.visual_encoder = visual_encoder
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        if self.full:
            inertial_encoder = BarlowModel().inertial_encoder
            inertial_encoder.load_state_dict(torch.load("/home/dfarkash/"+self.model_folder+"/inertial_encoder.pt"))
            inertial_encoder.eval()
            self.inertial_encoder = inertial_encoder

            for param in self.inertial_encoder.parameters():
                param.requires_grad = False

        preferences_path = "/home/dfarkash/cost_data/preferences.pkl"
        cprint('Loading user preferences from {}'.format(preferences_path))
        self.preferences = pickle.load(open(preferences_path, 'rb'))

        self.preferences = [0,6,1,2,5,4,3]

        model_path = "/home/dfarkash/"+self.model_folder+"/model.pkl"
        cprint('Loading cluster_model from {}'.format(model_path))
        self.model = pickle.load(open(model_path, 'rb'))

        self.cost_net = nn.Sequential(
            nn.Linear(128, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, 1), nn.ReLU(inplace=True)
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)
  
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, patch):
        with torch.no_grad():
            v_encoded = self.visual_encoder(patch)

        cost = self.cost_net(v_encoded)
        
  
        return cost



    def find_true_cost(self, patch, inertial):
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

        scaler = StandardScaler()
        representation=representation.cpu()
        representation=representation.detach().numpy()
        # representation = scaler.fit_transform(representation)

        # print(representation)

        cluster = self.model.predict(representation)

        # print(cluster)

        cluster = cluster.astype('int')

        true_cost = []
        
        for i in range(len(cluster)):
            true_cost.append(self.preferences[cluster[i]])

        true_cost = torch.Tensor(true_cost).cuda()

        return true_cost



    def smax(self, s1, s2):
        # return torch.pow((torch.exp(s1) / (torch.exp(s1) + torch.exp(s2))) - 1/2 , 2)
        return (torch.exp(s1) / (torch.exp(s1) + torch.exp(s2))) 

    def training_step(self, batch, batch_idx):
        self.visual_encoder.eval()
        if self.full:
            self.inertial_encoder.eval()
        self.cost_net.train()

        patch, inertial = batch

        true_cost = self.find_true_cost(patch, inertial)

        # print(true_cost)

        cost = self(patch)


        model = False
        if model:
            patch, true_cost = batch
            cost = self(patch)
            # cost = torch.reshape(cost,(256,1))
            # true_cost = torch.reshape(true_cost,(256,1))

        linear = False

        if linear :
            mse = torch.nn.MSELoss()
            # print("c"+str(cost))
            # print("t"+str(true_cost))
            loss = mse(cost[:,0], true_cost)
            # loss= loss.item()
            # loss=loss.float()
            # print(loss)
        else: 
            loss_lst = torch.zeros(len(true_cost)-1)
            i = 0
            while i < len(true_cost)-1:
                if true_cost[i] < true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i,0], cost[i+1,0])
                elif true_cost[i] > true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i+1,0], cost[i,0])
                else:
                    loss_lst[i] = (true_cost[i] - true_cost[i+1])**2
                    # loss_lst[i] = -1
                # penalize loss from going above 100.0
                if cost[i,0] > 100.0:
                    loss_lst[i] += (cost[i, 0]- 100.0)**2
                i=i+1

            loss = torch.mean(loss_lst)
            # loss= torch.mean(loss_lst[loss_lst != -1])

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.visual_encoder.eval()
        if self.full:
            self.inertial_encoder.eval()
        self.cost_net.eval()

        patch, inertial = batch

        true_cost = self.find_true_cost(patch, inertial)

        # print(true_cost)

        cost = self(patch)


        model = False
        if model:
            patch, true_cost = batch
            cost = self(patch)
            # cost = torch.reshape(cost,(256,1))
            # true_cost = torch.reshape(true_cost,(256,1))
            
        linear = False

        if linear :
            mse = torch.nn.MSELoss()
            # print("c"+str(cost))
            # print("t"+str(true_cost))
            loss = mse(cost[:,0], true_cost)
            # loss= loss.item()
            # loss=loss.float()
            # print(loss)
        else: 
            loss_lst = torch.zeros(len(true_cost)-1)
            i = 0
            while i < len(true_cost)-1:
                if true_cost[i] < true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i,0], cost[i+1,0])
                elif true_cost[i] > true_cost[i+1]:
                    loss_lst[i] = self.smax(cost[i+1,0], cost[i,0])
                else:
                    loss_lst[i] = (true_cost[i] - true_cost[i+1])**2
                    # loss_lst[i] = -1
                # penalize loss from going above 100.0
                if cost[i,0] > 100.0:
                    loss_lst[i] += (cost[i, 0]- 100.0)**2
                # loss_lst[i] += torch.max(torch.zeros(1).to(self.device), (cost[i, 0]- 100.0))**2
                i=i+1

            loss = torch.mean(loss_lst)
            # loss= torch.mean(loss_lst[loss_lst != -1])

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        # scheduler = CosineAnnealingLR(optimizer, T_max = 15000, eta_min = 0.00001)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=60, warmup_start_lr = 3e-4, eta_min = 1e-5)

        # return [optimizer], [scheduler]
        return optimizer
    

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        # if self.current_epoch % 2 == 0 and torch.cuda.current_device() == 0 :
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

    def scatter(self, true_cost, cost, vis_patch):

        # tc = torch.Tensor.cpu(true_cost).detach().numpy()
        # c = torch.Tensor.cpu(cost).detach().numpy()
        tc = true_cost
        c = cost

        dict = {}
        num_bins = 100
        min_size = 0
        max_size = 40
        bin_size = (max_size - min_size)/num_bins

        for i in range(num_bins):
            dict[i] = []
        dict["sizes"] = []
        for j in range(len(tc)):
            bin = int(c[j]/bin_size)
            dict[bin].append((tc[j],vis_patch[j,:,:,:],c[j]))

        for k in range (num_bins):
            dict["sizes"].append(len(dict[k]))

        max_size = max(dict["sizes"])

        new_mat = torch.zeros((len(tc), 3))
        new_true_cost = torch.zeros(len(tc))
        new_cost = torch.zeros(len(tc))
        new_vis_patch = torch.zeros((len(tc),3,64,64))
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

        metadata = list(zip(new_cost, new_true_cost))

        metadata_header = ["costs","true_costs"]

        self.logger.experiment.add_embedding(new_mat,
                    label_img=new_vis_patch,
                    global_step=self.current_epoch,
                    metadata=metadata,
                    metadata_header = metadata_header)
        # add the embedding to tensorboard

        
        


    def on_validation_end(self) -> None:
        # if not self.visual_patch: return
        if self.current_epoch % 2 == 0 and torch.cuda.current_device() == 0:
            self.patch = torch.cat(self.patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            idx = np.arange(self.visual_encoding.shape[0])

            np.random.shuffle(idx)


            ve = self.visual_encoding[idx,:]

            # print(ve.shape)

            vis_patch = self.patch[idx,:,:,:]
           
            cost = self.cost[idx]

            # print("cost shape: ", cost.shape)
            # print(cost)

            cost_frame = pd.DataFrame(cost[:,0])

            # print(cost)
            breaks = jenkspy.jenks_breaks(cost_frame[0] ,nb_class=7) 
            cost_bin = pd.cut(cost_frame[0], bins=breaks, labels=False, include_lowest=True)

            true_cost = self.true_cost[idx]

            acc = self.accuracy_naive(cost_bin, true_cost)

            self.logger.experiment.add_scalar("Binned accuracy", acc, self.current_epoch)

            if False:
                metadata = list(zip(cost[:2000], true_cost[:2000],cost_bin[:2000]))

                metadata_header = ["costs","true_costs","cost bins"]

                mat=ve[:2000]
                # mat = self.scatter(cost[:500],true_cost[:500])
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
    
    # print("main")
    # print(args.full)
    # check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = s2DataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size,
                        full = bool(args.full))

    model = s2Model(lr=args.lr, latent_size=args.latent_size,
                        inertial_shape=1200, scale_loss=1.0, lambd=1./args.latent_size, 
                          per_device_batch_size=args.batch_size, full=bool(args.full), 
                          model_folder = args.model_folder).to(device)


    early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
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

    trainer.fit(model, dm)