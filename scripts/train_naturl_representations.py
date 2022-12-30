#!/usr/bin/env python3

"""code to train the representation learning from the spot data"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pickle
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
import argparse
import yaml
import os

from scipy.signal import periodogram
from scripts.models import ProprioceptionModel, VisualEncoderModel
from scripts.utils import process_feet_data

import albumentations as A
from torchvision import transforms

import tensorboard as tb

from termcolor import cprint

from scripts import cluster_jackal

terrain_label = {
    'cement': 0,
    'pebble_pavement': 1,
    'grass': 2,
    'dark_tile': 3,
    'bush': 4,
    'asphalt': 5,
    'marble_rock': 6,
    'red_brick': 7, 
}

FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0


class TerrainDataset(Dataset):
    def __init__(self, pickle_files_root, incl_orientation=False, mean=None, std=None, train=False):
        self.pickle_files_paths = glob.glob(pickle_files_root + '/*.pkl')
        self.label = pickle_files_root.split('/')[-2]
        # self.label = terrain_label[self.label]
        self.incl_orientation = incl_orientation
        self.mean, self.std = mean, std
        if train:
            # cprint('Using data augmentation', 'green')
            # use albumentation for data augmentation
            self.transforms = A.Compose([
                A.Flip(always_apply=False, p=0.5),
                A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.2, 0.3), rotate_limit=(-21, 21), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
                A.RandomBrightness(always_apply=False, p=0.5, limit=(-0.2, 0.2)),
            ])
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.pickle_files_paths)
    
    def __getitem__(self, idx):
        with open(self.pickle_files_paths[idx], 'rb') as f:
            data = pickle.load(f)
        patches, imu, feet, leg = data['patches'], data['imu'], data['feet'], data['leg']
        
        # process the feet data to remove the mu and std values for non-contacting feet
        feet = process_feet_data(feet)
        
        if not self.incl_orientation: imu = imu[:, :-3]
        
        # normalize the imu data
        if self.mean is not None and self.std is not None:
            imu = (imu - self.mean['imu']) / (self.std['imu'] + 1e-7)
            imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
            imu = imu.flatten()
            imu = imu.reshape(1, -1)
            
            leg = (leg - self.mean['leg']) / (self.std['leg'] + 1e-7)
            leg = periodogram(leg, fs=LEG_TOPIC_RATE, axis=0)[1]
            leg = leg.flatten()
            leg = leg.reshape(1, -1)
            
            feet = (feet - self.mean['feet']) / (self.std['feet'] + 1e-7)
            feet = periodogram(feet, fs=FEET_TOPIC_RATE, axis=0)[1]
            feet = feet.flatten()
            feet = feet.reshape(1, -1)
        
        # sample 2 values between 0 and num_patches-1
        patch_1_idx, patch_2_idx = np.random.choice(len(patches), 2, replace=False)
        patch1, patch2 = patches[patch_1_idx], patches[patch_2_idx]
        
        # convert BGR to RGB
        patch1, patch2 = patch1[:, :, ::-1], patch2[:, :, ::-1]
        
        # apply the transforms
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)['image']
            patch2 = self.transforms(image=patch2)['image']
        
        # normalize the image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0
        
        # transpose
        patch1, patch2 = np.transpose(patch1, (2, 0, 1)), np.transpose(patch2, (2, 0, 1))
        
        return np.asarray(patch1), np.asarray(patch2), imu, leg, feet, self.label

# create pytorch lightning data module
class NATURLDataModule(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=64, num_workers=2, include_orientation_imu=False):
        super().__init__()
        
        # read the yaml file
        cprint('Reading the yaml file at : {}'.format(data_config_path), 'green')
        self.data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        self.data_config_path = '/'.join(data_config_path.split('/')[:-1])

        self.include_orientation_imu = include_orientation_imu

        self.batch_size, self.num_workers = batch_size, num_workers
        
        self.mean, self.std = {}, {}
        
        # load the train and val datasets
        self.load()
        cprint('Train dataset size : {}'.format(len(self.train_dataset)), 'green')
        cprint('Val dataset size : {}'.format(len(self.val_dataset)), 'green')
        
        
    def load(self):
        
        # check if the data_statistics.pkl file exists
        if os.path.exists(self.data_config_path + '/data_statistics.pkl'):
            cprint('Loading the mean and std from the data_statistics.pkl file', 'green')
            data_statistics = pickle.load(open(self.data_config_path + '/data_statistics.pkl', 'rb'))
            self.mean, self.std = data_statistics['mean'], data_statistics['std']
            
        else:
            # find the mean and std of the train dataset
            cprint('Finding the mean and std of the train dataset', 'green')
            self.tmp_dataset = ConcatDataset([TerrainDataset(pickle_files_root, incl_orientation=self.include_orientation_imu) for pickle_files_root in self.data_config['train']])
            self.tmp_dataloader = DataLoader(self.tmp_dataset, batch_size=1, num_workers=0, shuffle=False)
            # find the mean and std of the train dataset
            imu_data, leg_data, feet_data = [], [], []
            for _, _, imu, leg, feet, _ in self.tmp_dataloader:
                imu_data.append(imu.cpu().numpy())
                leg_data.append(leg.cpu().numpy())
                feet_data.append(feet.cpu().numpy())
            imu_data = np.concatenate(imu_data, axis=0)
            leg_data = np.concatenate(leg_data, axis=0)
            feet_data = np.concatenate(feet_data, axis=0)
            
            imu_data = imu_data.reshape(-1, imu_data.shape[-1])
            leg_data = leg_data.reshape(-1, leg_data.shape[-1])
            feet_data = feet_data.reshape(-1, feet_data.shape[-1])
            
            self.mean['imu'], self.std['imu'] = np.mean(imu_data, axis=0), np.std(imu_data, axis=0)
            self.mean['leg'], self.std['leg'] = np.mean(leg_data, axis=0), np.std(leg_data, axis=0)
            self.mean['feet'], self.std['feet'] = np.mean(feet_data, axis=0), np.std(feet_data, axis=0)
            
            cprint('Mean : {}'.format(self.mean), 'green')
            cprint('Std : {}'.format(self.std), 'green')
            
            # save the mean and std
            cprint('Saving the mean and std to the data_statistics.pkl file', 'green')
            pickle.dump({'mean': self.mean, 'std': self.std}, open(self.data_config_path + '/data_statistics.pkl', 'wb'))
            
        
        # load the train data
        self.train_dataset = ConcatDataset([TerrainDataset(pickle_files_root, incl_orientation=self.include_orientation_imu, mean=self.mean, std=self.std, train=True) for pickle_files_root in self.data_config['train']])
        self.val_dataset = ConcatDataset([TerrainDataset(pickle_files_root, incl_orientation=self.include_orientation_imu, mean=self.mean, std=self.std) for pickle_files_root in self.data_config['val']])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last= True if len(self.train_dataset) % self.batch_size != 0 else False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last= True if len(self.val_dataset) % self.batch_size != 0 else False)


class NATURLRepresentationsModel(pl.LightningModule):
    def __init__(self, lr=3e-4, latent_size=64, scale_loss=1.0/32, lambd=3.9e-6, weight_decay=1e-6, l1_coeff=0.5):
        super(NATURLRepresentationsModel, self).__init__()
        
        self.save_hyperparameters(
            'lr',
            'latent_size',
            'scale_loss',
            'lambd',
            'weight_decay',
            'l1_coeff'
        )
        
        self.lr, self.latent_size, self.scale_loss, self.lambd, self.weight_decay = lr, latent_size, scale_loss, lambd, weight_decay
        self.l1_coeff = l1_coeff
        
        # visual encoder architecture
        self.visual_encoder = VisualEncoderModel(latent_size=128)
        
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=128)
        
        self.projector = nn.Sequential(
            nn.Linear(128, latent_size), nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size)
        )
        
        # coefficients for vicreg loss
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        
    
    def forward(self, patch1, patch2, inertial_data, leg, feet):
        v_encoded_1 = self.visual_encoder(patch1.float())
        v_encoded_2 = self.visual_encoder(patch2.float())
        
        # i_encoded = self.inertial_encoder(inertial_data.float())
        i_encoded = self.proprioceptive_encoder(inertial_data.float(), leg.float(), feet.float())
        
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        zi = self.projector(i_encoded)
        
        return zv1, zv2, zi, v_encoded_1, v_encoded_2, i_encoded
    
    def vicreg_loss(self, z1, z2):
        repr_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

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
        patch1, patch2, inertial, leg, feet, label = batch
        
        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)

        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label = batch
        
        # combine inertial and leg data
        # inertial = torch.cat((inertial, leg, feet), dim=-1)
        
        zv1, zv2, zi, _, _, _ = self.forward(patch1, patch2, inertial, leg, feet)
        
        # compute viewpoint invariance vicreg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        # compute visual-inertial vicreg loss
        loss_vi = 0.5 * self.vicreg_loss(zv1, zi) + 0.5 * self.vicreg_loss(zv2, zi)
        
        loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vpt_inv', loss_vpt_inv, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_vi', loss_vi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        # return torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        # save the batch data only every other epoch or during the last epoch
        if self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1:
            patch1, patch2, inertial, leg, feet, label = batch
            # combine inertial and leg data
            # inertial = torch.cat((inertial, leg, feet), dim=-1)
        
            with torch.no_grad():
                _, _, _, zv1, zv2, zi = self.forward(patch1, patch2, inertial, leg, feet)
            zv1, zi = zv1.cpu(), zi.cpu()
            patch1 = patch1.cpu()
            label = np.asarray(label)
            
            if batch_idx == 0:
                self.visual_encoding = [zv1]
                self.inertial_encoding = [zi]
                self.label = label
                self.visual_patch = [patch1]
            else:
                self.visual_encoding.append(zv1)
                self.inertial_encoding.append(zi)
                self.label = np.concatenate((self.label, label))
                self.visual_patch.append(patch1)
    
    def on_validation_end(self):
        if (self.current_epoch % 10 == 0 or self.current_epoch == self.trainer.max_epochs-1) and torch.cuda.current_device() == 0:
            self.visual_patch = torch.cat(self.visual_patch, dim=0)
            self.visual_encoding = torch.cat(self.visual_encoding, dim=0)
            self.inertial_encoding = torch.cat(self.inertial_encoding, dim=0)
            
            # randomize index selections
            idx = np.arange(self.visual_encoding.shape[0])
            np.random.shuffle(idx)
            
            # limit the number of samples to 2000
            ve = self.visual_encoding#[idx[:2000]]
            vi = self.inertial_encoding#[idx[:2000]]
            vis_patch = self.visual_patch#[idx[:2000]]
            ll = self.label#[idx[:2000]]
            
            data = torch.cat((ve, vi), dim=-1)
            
            # get results of k-means clustering
            # clusters, elbow, model = cluster_jackal.cluster_model(data)
            
            # calculate and print accuracy
            cprint('finding accuracy...', 'yellow')
            out = cluster_jackal.accuracy_naive(data, ll, label_types=list(terrain_label.keys()))
                
            # log k-means accurcay and projection for tensorboard visualization
            self.logger.experiment.add_scalar("K-means accuracy", out, self.current_epoch)
            
            if self.current_epoch % 10 == 0:
                self.logger.experiment.add_embedding(mat=data, label_img=vis_patch, global_step=self.current_epoch, metadata=ll, tag='visual_encoding')
            del self.visual_patch, self.visual_encoding, self.inertial_encoding, self.label
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dual Auto Encoder')
    parser.add_argument('--batch_size', '-b', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--l1_coeff', type=float, default=0.5, metavar='L1C',
                        help='L1 loss coefficient (1)')
    parser.add_argument('--num_gpus','-g', type=int, default=2, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--save', type=int, default=0, metavar='N',
                        help='Whether to save the k means model and encoders at the end of the run')
    parser.add_argument('--imu_in_rep', type=int, default=1, metavar='N',
                        help='Whether to include the inertial data in the representation')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    args = parser.parse_args()
    
    model = NATURLRepresentationsModel(lr=args.lr, latent_size=args.latent_size, l1_coeff=args.l1_coeff)
    dm = NATURLDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    
    early_stopping_cb = EarlyStopping(monitor='train_loss', mode='min', min_delta=0.00, patience=1000)
    # create model checkpoint only at end of training
    model_checkpoint_cb = ModelCheckpoint(dirpath='models/', filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_', verbose=True)
    
    print("Training model...")
    trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
                         max_epochs=args.epochs,
                        #  callbacks=[model_checkpoint_cb],
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         logger=True,
                         sync_batchnorm=True,
                         )

    # fit the model
    trainer.fit(model, dm)
    
    # save the models as .pt files
    # save visual encoder
    torch.save(model.visual_encoder.state_dict(), 'models/visual_encoder.pt')
    # save proprioceptive encoder
    torch.save(model.proprioceptive_encoder.state_dict(), 'models/inertial_encoder.pt')
    
    
    

