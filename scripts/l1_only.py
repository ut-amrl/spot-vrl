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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cluster_jackal
from dict_custom_data import CustomDataset, MyDataLoader

def exclude_bias_and_norm(p):
    return p.ndim == 1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 2, 2)

class BarlowModel(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=64, inertial_shape=None,
                 scale_loss:float=1.0/32, lambd:float=3.9e-6, weight_decay=1e-6,
                 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1
                 , l1_coeff = 1):
        super(BarlowModel, self).__init__()

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

        # self.visual_encoder = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False), # 63 x 63
        #     nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), # 31 x 31
        #     nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False), # 14 x 14
        #     nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),  # 5 x 5
        #     nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
        #     nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     Flatten(), # 256 output
        #     nn.Linear(256, 128)
        # )

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

        # self.inertial_encoder = nn.Sequential(
        #     nn.Conv1d(1, 8, kernel_size=3, stride=2, bias=False), nn.BatchNorm1d(8), nn.PReLU(),
        #     nn.AvgPool1d(kernel_size=3, stride=2),
        #     nn.Conv1d(8, 16, kernel_size=5, stride=2, bias=False), nn.BatchNorm1d(16), nn.PReLU(),
        #     nn.AvgPool1d(kernel_size=3, stride=2),
        #     nn.Conv1d(16, 32, kernel_size=5, stride=2, bias=False), nn.BatchNorm1d(32), nn.PReLU(),
        #     nn.AvgPool1d(kernel_size=3, stride=2),
        #     nn.Conv1d(32, 64, kernel_size=3, stride=2, bias=False), nn.BatchNorm1d(64), nn.PReLU(),
        #     nn.AvgPool1d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(256, 128)
        # )

        self.inertial_encoder = nn.Sequential(
            nn.Linear(600+6, 256, bias=False), nn.BatchNorm1d(256), nn.PReLU(),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 128)
        )

        self.projector = nn.Sequential(
            nn.Linear(128, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False), nn.BatchNorm1d(latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size, bias=False)
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(latent_size, affine=False, track_running_stats=False)
  
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, main_patch_lst, inertial_data, patch_list_1, patch_list_2):
        visual_patch = main_patch_lst[0]
        imu_history = inertial_data

        v_encoded = self.visual_encoder(visual_patch)
        i_encoded = self.inertial_encoder(imu_history)


        z1 = self.projector(v_encoded)
        z2 = self.projector(i_encoded)

        lst1_projected =[]
        lst2_projected =[]
  
        return z1, z2, lst1_projected, lst2_projected

    def barlow_loss(self, z1, z2):
        # z1 = (z1 - torch.mean(z1, dim=0))/(torch.std(z1, dim=0) + 1e-4)
        # z2 = (z2 - torch.mean(z2, dim=0))/(torch.std(z2, dim=0) + 1e-4)
  
        z1 = self.bn(z1)
        z2 = self.bn(z2)

        # empirical cross-correlation matrix
        c = z1.T @ z2
    
        # sum the cross-correlation matrix between all gpus
        c.div_(self.per_device_batch_size * self.trainer.num_processes)
        self.all_reduce(c)
    
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
  
        loss = on_diag + self.lambd * off_diag 
        return loss

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
        self.visual_encoder.train()
        self.inertial_encoder.train()
        self.projector.train()
        #try self.visual_encoder.train(), etc.
        main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch
        visual_emb, inertial_emb, lst1_emb, lst2_emb = self(main_patch_lst, inertial_data, patch_list_1, patch_list_2)
        # loss = self.barlow_loss(visual_emb, inertial_emb)
        l1 = self.vicreg_loss(visual_emb, inertial_emb)

        loss = l1
        # loss=l2
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.visual_encoder.eval()
        self.inertial_encoder.eval()
        self.projector.eval()
        main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch
        visual_emb, inertial_emb, lst1_emb, lst2_emb = self(main_patch_lst, inertial_data, patch_list_1, patch_list_2)
        # loss = self.barlow_loss(visual_emb, inertial_emb)
        l1 = self.vicreg_loss(visual_emb, inertial_emb)
        

        loss = l1
        # loss=l2
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        # scheduler = CosineAnnealingLR(optimizer, T_max = 15000, eta_min = 0.00001)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=60, warmup_start_lr = 3e-4, eta_min = 1e-5)

        # return [optimizer], [scheduler]
        return optimizer
    
    # def configure_optimizers(self):
    # 	optimizer = LARS(
    # 		self.parameters(),
    # 		lr=0,  # Initialize with a LR of 0
    # 		weight_decay=self.weight_decay,
    # 		weight_decay_filter=exclude_bias_and_norm,
    # 		lars_adaptation_filter=exclude_bias_and_norm
    # 	)

    # 	total_training_steps = self.total_training_steps
    # 	num_warmup_steps = self.compute_warmup(total_training_steps, self.num_warmup_steps_or_ratio)
    # 	lr_scheduler = CosineWarmupScheduler(
    # 		optimizer=optimizer,
    # 		batch_size=self.per_device_batch_size,
    # 		warmup_steps=num_warmup_steps,
    # 		max_steps=total_training_steps,
    # 		lr=self.lr
    # 	)
    # 	return [optimizer], [
    # 		{
    # 			'scheduler': lr_scheduler,  # The LR scheduler instance (required)
    # 			'interval': 'step',  # The unit of the scheduler's step size
    # 		}
    # 	]

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.current_epoch % 2 == 0 :

            main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = batch
            visual_patch = main_patch_lst[0]
            label = np.asarray(label) 
            visual_patch = visual_patch.float()
            inertial_data = inertial_data.float()
            with torch.no_grad():
                visual_encoding = self.visual_encoder(visual_patch.cuda())
                inertial_encoding = self.inertial_encoder(inertial_data.cuda())
            visual_encoding = visual_encoding.cpu()
            inertial_encoding = inertial_encoding.cpu()
            visual_patch = visual_patch.cpu()
               # visual_encoding = F.normalize(visual_encoding, dim=1)
            
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

    def sample_clusters(data, visual_patch):
            clusters , elbow = cluster_jackal.cluster(data)
            dic = {}
            for i in range(elbow):
                dic[i] = []
            for i in range(elbow):
                idx = np.where(clusters ==i)
                for j in range(25):
                    chosen = np.random.randint(0, len(idx))
                    visual_patch = visual_patch[idx[chosen], :, :, :]
                    dic[i].append(visual_patch)

            return dic, elbow

    def img_clusters(dic, elbow):
        for i in range(elbow):
            new_im = Image.new('RGB', (3000,3000))
            for j in range(25):
                visual_patch = dic[i][j]
                visual_patch = visual_patch.cpu()
                visual_patch = visual_patch.numpy()
                h = int(j/5)
                w = j%5
                im = im.fromarray(visual_patch)
                im.thumbnail((300,300))
                new_im.paste(im, (h,w))
            new_im.save("/home/dfarkash/garbage" +"/group"+str(j)+".png")

    def on_validation_end(self) -> None:
        # if not self.visual_patch: return
        if self.current_epoch % 2 == 0:
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            idx = np.arange(self.visual_encoding.shape[0])

            # randomize numpy array
            np.random.shuffle(idx)

            # clusters , elbow = cluster_jackal.cluster(self.visual_encoding[idx[:2000],:])
            # metadata = list(zip(self.label[idx[:2000]], clusters))


            # metadata_header = ["labels","clusters"]
            # out = cluster_jackal.accuracy_naive(self.visual_encoding[idx[:2000],:],self.label[idx[:2000]])

            # self.logger.experiment.add_scalar("K-means accuracy", out, self.current_epoch)

            # self.logger.experiment.add_embedding(mat=self.visual_encoding[idx[:2000], :],
            #                                      label_img=self.visual_patch[idx[:2000], :, :, :],
            #                                      global_step=self.current_epoch,
            #                                      metadata=metadata,
            #                                      metadata_header = metadata_header
            #                                     )
            # del self.visual_patch, self.visual_encoding, self.label

            ve = self.visual_encoding[idx[:2000],:]
            # print(ve.size())
            ie = self.inertial_encoding[idx[:2000],:]
            # print(ie.size())
            data = torch.cat((ve, ie), dim=1)
            # print(data.size())

            clusters , elbow = cluster_jackal.cluster(data)
            metadata = list(zip(self.label[idx[:2000]], clusters))


            metadata_header = ["labels","clusters"]
            out = cluster_jackal.accuracy_naive(data,self.label[idx[:2000]])

            self.logger.experiment.add_scalar("K-means accuracy", out, self.current_epoch)

            self.logger.experiment.add_embedding(mat=data,
                                                 label_img=self.visual_patch[idx[:2000], :, :, :],
                                                 global_step=self.current_epoch,
                                                 metadata=metadata,
                                                 metadata_header = metadata_header
                                                )
            

            # self.img_clusters(self.sample_clusters(data, self.visual_patch[idx[:2000], :, :, :]))

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
    parser.add_argument('--dataset_config_path', type=str, default='jackal_data/dataset_config_haresh_local.yaml')
    args = parser.parse_args()
    
    # check if the dataset config yaml file exists
    if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = MyDataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size)

    model = BarlowModel(lr=args.lr, latent_size=args.latent_size,
                        inertial_shape=1200, scale_loss=1.0, lambd=1./args.latent_size, 
                          per_device_batch_size=args.batch_size, l1_coeff = args.l1_coeff).to(device)

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

    # trainer = pl.Trainer(
    #                      devices=1,
    #                      max_epochs=args.epochs,
    #                      callbacks=[model_checkpoint_cb],
    #                      log_every_n_steps=10,
    #                      strategy='ddp',
    #                      num_sanity_val_steps=0,
    #                      logger=True,
    #                      )

    trainer.fit(model, dm)