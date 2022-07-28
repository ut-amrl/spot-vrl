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
                 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1):
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

        # net_path = "/home/dfarkash/spot-vrl/models/26-07-2022-11-06-47_.ckpt"
        # cprint('Loading model from {}'.format(net_path))
        # net = BarlowModel.load_from_checkpoint(net_path)
        # net.eval()
        # self.visual_encoder = net.visual_encoder

        visual_encoder = BarlowModel().visual_encoder
        visual_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/visual_encoder.pt"))
        visual_encoder.eval()
        self.visual_encoder = visual_encoder
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        inertial_encoder = BarlowModel().inertial_encoder
        inertial_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/inertial_encoder.pt"))
        inertial_encoder.eval()
        self.inertial_encoder = inertial_encoder

        for param in self.inertial_encoder.parameters():
            param.requires_grad = False

        preferences_path = "/home/dfarkash/cost_data/preferences.pkl"
        cprint('Loading user preferences from {}'.format(preferences_path))
        self.preferences = pickle.load(open(preferences_path, 'rb'))

        self.preferences = [0,5,10,15,20,25,30]

        model_path = "/home/dfarkash/cost_data/model.pkl"
        cprint('Loading cluster_model from {}'.format(model_path))
        self.model = pickle.load(open(model_path, 'rb'))

        self.cost_net = nn.Sequential(
            nn.Linear(128, latent_size, bias=False), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False), nn.ReLU(inplace=True),
            nn.Linear(latent_size, 1, bias=False)
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)
  
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, patch):

        v_encoded = self.visual_encoder(patch)

        cost = self.cost_net(v_encoded)
        
  
        return cost



    def find_true_cost(self, patch, inertial):
        self.visual_encoder.eval()
        self.inertial_encoder.eval()
        
        with torch.no_grad():
            visual_encoding = self.visual_encoder(patch)
            inertial_encoding = self.inertial_encoder(inertial)

        representation = torch.cat((visual_encoding, inertial_encoding), dim=1)

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

    def training_step(self, batch, batch_idx):
        self.visual_encoder.eval()
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

        mse = torch.nn.MSELoss()
        # print("c"+str(cost))
        # print("t"+str(true_cost))
        loss = mse(cost[:,0], true_cost)
        # loss= loss.item()
        # loss=loss.float()
        # print(loss)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.visual_encoder.eval()
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
            
        mse = torch.nn.MSELoss()
        # print("c"+str(cost))
        # print("t"+str(true_cost))
        loss = mse(cost[:,0], true_cost)
        # loss= loss.item()
        # loss=loss.float()
        # print(loss)

        # print(cost[:,0])
        # print(true_cost)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        # scheduler = CosineAnnealingLR(optimizer, T_max = 15000, eta_min = 0.00001)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=60, warmup_start_lr = 3e-4, eta_min = 1e-5)

        # return [optimizer], [scheduler]
        return optimizer
    

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.current_epoch % 2 == 0 :

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


    def on_validation_end(self) -> None:
        # if not self.visual_patch: return
        if self.current_epoch % 2 == 0:
            self.patch = torch.cat(self.patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            idx = np.arange(self.visual_encoding.shape[0])

            np.random.shuffle(idx)


            ve = self.visual_encoding[idx[:2000],:]

            vis_patch = self.patch[idx[:2000],:,:,:]
           
            cost = self.cost[idx[:2000]]

            true_cost = self.true_cost[idx[:2000]]

            metadata = list(zip(cost, true_cost))


            metadata_header = ["costs","true_costs"]

            self.logger.experiment.add_embedding(mat=ve,
                                                 label_img=vis_patch,
                                                 global_step=self.current_epoch,
                                                 metadata=metadata,
                                                metadata_header=metadata_header)
                                               

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
    parser.add_argument('--num_gpus', type=int, default=3, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=256, metavar='N',
                        help='Size of the common latent space (default: 512)')
    parser.add_argument('--dataset_config_path', type=str, default='jackal_data/dataset_config_haresh_local.yaml')
    args = parser.parse_args()
    
    # check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = s2DataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size)

    model = s2Model(lr=args.lr, latent_size=args.latent_size,
                        inertial_shape=1200, scale_loss=1.0, lambd=1./args.latent_size, 
                          per_device_batch_size=args.batch_size).to(device)

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
                         )

    trainer.fit(model, dm)