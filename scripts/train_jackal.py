import glob

from matplotlib.pyplot import axes
import pytorch_lightning as pl
import torch
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
import copy
import random
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from typing import List, Union, Tuple
import os
import yaml

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def exclude_bias_and_norm(p):
	return p.ndim == 1

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
	def forward(self, input, size=256):
		return input.view(input.size(0), size, 2, 2)

class CustomDataset(Dataset):
	def __init__(self, pickle_file_path):
		self.pickle_file_path = pickle_file_path
		cprint('Loading data from {}'.format(pickle_file_path))
		self.data = pickle.load(open(self.pickle_file_path, 'rb'))
		self.label = pickle_file_path.split('/')[-2]

	def __len__(self):
		return len(self.data['patches'])

	def __getitem__(self, idx):
		# randomly pick a patch from the list
		patch_1 = copy.deepcopy(random.sample(self.data['patches'][idx], 1)[0])
		# patch_1 = self.data['patches'][idx][0]
  
		patch_1 = cv2.resize(patch_1, (128, 128))
		patch_1 = patch_1.astype(np.float32) / 255.0 # normalize
		patch_1 = np.moveaxis(patch_1, -1, 0)
  
		patch_2 = copy.deepcopy(random.sample(self.data['patches'][idx], 1)[0])
		patch_2 = cv2.resize(patch_2, (128, 128))
		patch_2 = patch_2.astype(np.float32) / 255.0 # normalize
		patch_2 = np.moveaxis(patch_2, -1, 0)

		inertial_data = copy.deepcopy(self.data['imu_jackal'][idx])
		# inertial_data = np.expand_dims(inertial_data, axis=0)
		
		inertial_data = inertial_data.reshape(200, 6, 1)
		inertial_data = cv2.resize(inertial_data, (128, 128))
		inertial_data = inertial_data.reshape(1, 128, 128)

		return patch_1, patch_2, inertial_data, self.label

class CustomExpandedDataset(Dataset):
	def __init__(self, pickle_file_path):
		self.pickle_file_path = pickle_file_path
		cprint('Loading data from {}'.format(pickle_file_path))
		self.data = pickle.load(open(self.pickle_file_path, 'rb'))
		self.label = pickle_file_path.split('/')[-2]

		self.patch, self.inertial  = [], []
		for i in range(len(self.data['patches'])):
			self.patch.extend(self.data['patches'][i])
			self.inertial.extend([self.data['imu_jackal'][i] for _ in range(len(self.data['patches'][i]))])

	def __len__(self):
		return len(self.inertial)

	def __getitem__(self, idx):
		patch_1 = self.patch[idx]
		patch_1 = cv2.resize(patch_1, (128, 128))
		patch_1 = patch_1.astype(np.float32) / 255.0 # normalize
		patch_1 = np.moveaxis(patch_1, -1, 0)
  
		inertial_data = self.inertial[idx]

		return patch_1, torch.ones(1), inertial_data, self.label

class MyDataLoader(pl.LightningDataModule):
	def __init__(self, data_config_path, batch_size=32):
		super(MyDataLoader, self).__init__()
		self.batch_size = batch_size
		self.data_config_path = data_config_path
		self.setup()

	def setup(self, stage=None):
		with open(self.data_config_path) as file:
			data_config = yaml.full_load(file)
   
		train_data_path = data_config['train']
		val_data_path = data_config['val']

		self.train_dataset = ConcatDataset([CustomDataset(file) for file in train_data_path])
		self.val_dataset = ConcatDataset([CustomDataset(file) for file in val_data_path])

		# find mean, std statistics of inertial data in the training set
		print('Finding mean and std statistics of inertial data in the training set...')
		tmp = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
		for _,_, i, _ in tmp:
			i = i.numpy()
			break
		self.mean = None
		self.std = None
		self.inertial_shape = i.shape[1]
		print('Inertial shape:', self.inertial_shape)
		print('Data statistics have been found.')
		del tmp

		print('Train dataset size:', len(self.train_dataset))
		print('Val dataset size:', len(self.val_dataset))

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20, drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20, drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)

class BarlowModel(pl.LightningModule):
	def __init__(self, lr=3e-4, latent_size=64, inertial_shape=None, lambd:float=3.9e-6, weight_decay=1.5e-6,
				 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1):
		super(BarlowModel, self).__init__()

		self.save_hyperparameters(
			'lr',
			'latent_size',
			'inertial_shape',
			'lambd',
			'weight_decay'
		)
  
		self.best_val_loss = None
		self.curr_val_loss, self.val_steps = 0.0, 0
		self.curr_train_loss, self.train_steps = 0.0, 0

		self.lambd = lambd
		self.weight_decay = weight_decay
		self.lr = lr
		self.per_device_batch_size = per_device_batch_size
		self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio
		self.idnt = torch.eye(latent_size).to(self.device)
		self.off_diag = torch.mul(torch.ones((latent_size, latent_size), dtype=bool).to(self.device), self.lambd).fill_diagonal_(1)

		

		# self.visual_encoder = nn.Sequential(
		# 	nn.Conv2d(3, 16, kernel_size=3, stride=2), # 63 x 63
		# 	nn.ReLU(),
		# 	nn.Conv2d(16, 32, kernel_size=3, stride=2), # 31 x 31
		# 	nn.ReLU(),
		# 	nn.Conv2d(32, 64, kernel_size=5, stride=2), # 14 x 14
		# 	nn.ReLU(),
		# 	nn.Conv2d(64, 128, kernel_size=5, stride=2),  # 5 x 5
		# 	nn.ReLU(),
		# 	nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
		# 	nn.ReLU(),
		# 	Flatten(), # 1024 output
		# 	nn.Linear(1024, 64)
		# )
  
		# self.inertial_encoder = nn.Sequential(
		# 	nn.Linear(inertial_shape, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
		# 	nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
		# 	nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
		# 	nn.Linear(256, 128)
		# )
  
		self.visual_encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=2, bias=False), nn.InstanceNorm2d(16), nn.ReLU(inplace=True),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=2, stride=2),
			Flatten(), # 256 output
			nn.Linear(256, 128), nn.PReLU(),
			nn.Linear(128, 64) # 64 output
		)
  
		self.inertial_encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=2, bias=False), nn.InstanceNorm2d(16), nn.ReLU(inplace=True),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=2, stride=2),
			Flatten(), # 256 output
			nn.Linear(256, 128), nn.PReLU(),
			nn.Linear(128, 64) # 64 output
		)

		# self.projectorv = nn.Sequential(
		# 	nn.Linear(64, 256), nn.ReLU(inplace=True),
		# 	nn.Linear(256, 256), nn.ReLU(inplace=True),
		# 	nn.Linear(256, latent_size, bias=False)
		# )
  
		# self.projectori = nn.Sequential(
		# 	nn.Linear(64, 256), nn.ReLU(inplace=True),
		# 	nn.Linear(256, 256), nn.ReLU(inplace=True),
		# 	nn.Linear(256, latent_size, bias=False)
		# )
  
		self.projector = nn.Sequential(
			nn.Linear(64, 256), nn.ReLU(inplace=True),
			nn.Linear(256, 256), nn.ReLU(inplace=True),
			nn.Linear(256, latent_size, bias=False)
		)

		# self.bn = nn.BatchNorm1d(latent_size, affine=False)
  
		self.sim_coeff = 25.0
		self.std_coeff = 25.0
		self.cov_coeff = 1.0
	
	def forward(self, visual_patch, imu_history):
		v_encoded = self.visual_encoder(visual_patch)
		i_encoded = self.inertial_encoder(imu_history)
  
		# L2 normalize along encoding dimension
		# v_encoded = F.normalize(v_encoded, dim=1)
		# i_encoded = F.normalize(i_encoded, dim=1)
	
		z1 = self.projectorv(v_encoded)
		z2 = self.projectori(i_encoded)
		return z1, z2  

	def forward_vision_only(self, visual_patch_1, visual_patch_2):
		return self.projector(self.visual_encoder(visual_patch_1)), self.projector(self.visual_encoder(visual_patch_2))

	def barlow_loss(self, z1, z2):
		z1 = (z1 - torch.mean(z1, dim=0))/(torch.std(z1, dim=0) + 1e-6)
		z2 = (z2 - torch.mean(z2, dim=0))/(torch.std(z2, dim=0) + 1e-6)
		# z1, z2 = self.bn(z1), self.bn(z2)
  
		N = z1.shape[0]
	
		# empirical cross-correlation matrix
		# c = self.bn(z1).T @ self.bn(z2)
		c = z1.T @ z2
		c.div_(N)

		# c1b = (z1.T).T @ z1.T / N
		# c2b = (z2.T).T @ z2.T / N

		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
		off_diag = self.off_diagonal(c).pow_(2).sum()
		# off_diag_match_loss = (c1b - c2b).pow_(2).sum()
		# off_diag_match_loss = F.cosine_similarity(c1b.view((c1b.shape[0], -1)), c2b.view((c2b.shape[0], -1))).sum()
  
		loss = on_diag + self.lambd * off_diag #+ self.lambd * off_diag_match_loss
		return loss, on_diag, self.lambd * off_diag

	def vicreg_loss(self, z1, z2):
		repr_loss = F.mse_loss(z1, z2)

		z1 = z1 - z1.mean(dim=0)
		z2 = z2 - z2.mean(dim=0)
  
		std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
		std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
		std_loss = torch.mean(F.relu(1 - std_z1)) / 2 + torch.mean(F.relu(1 - std_z2)) / 2

		cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
		cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
		cov_loss = off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + off_diagonal(cov_y).pow_(2).sum().div_(z2.shape[1])
  
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
		visual, visual2, inertial, _ = batch
		# z1, z2 = self.forward(visual, inertial)
		z1, z2 = self.forward_vision_only(visual, visual2)
  
		# loss, on_diag, off_diag = self.barlow_loss(z1, z2)
		loss = self.vicreg_loss(z1, z2)
		# self.curr_train_loss = loss.item()
		# self.train_steps += 1
		self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
		# self.log('train_on_diag', on_diag, prog_bar=False, logger=True, on_epoch=True, on_step=False)
		# self.log('train_off_diag', off_diag, prog_bar=False, logger=True, on_epoch=True, on_step=False)
		return loss

	def validation_step(self, batch, batch_idx):
		visual, visual2, inertial, _ = batch
		# z1, z2 = self.forward(visual, inertial)
		z1, z2 = self.forward_vision_only(visual, visual2)
  
		# z1, z2 = self.forward_vision_only(visual, visual2)
		# loss, on_diag, off_diag = self.barlow_loss(z1, z2)
		loss = self.vicreg_loss(z1, z2)
		# self.curr_val_loss += loss.item()
		# self.val_steps += 1
		self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
		# self.log('val_on_diag', on_diag, prog_bar=False, logger=True, on_epoch=True, on_step=False)
		# self.log('val_off_diag', off_diag, prog_bar=False, logger=True, on_epoch=True, on_step=False)
		return loss

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if self.current_epoch % 10 == 0:

			visual_patch, visual_patch_2, imu_history, label = batch
			label = np.asarray(label)
			visual_patch = visual_patch.float()
			with torch.no_grad():
				visual_encoding = self.visual_encoder(visual_patch.to(self.device))
				# visual_encoding = F.normalize(visual_encoding, dim=1)
			visual_encoding = visual_encoding.cpu()
			visual_patch = visual_patch.cpu()

			if batch_idx == 0:
				self.visual_patch = visual_patch[:, :, :, :]
				self.visual_encoding = visual_encoding[:, :]
				self.label = label[:]
			else:
				self.visual_patch = torch.cat((self.visual_patch, visual_patch[:, :, :, :]), dim=0)
				self.visual_encoding = torch.cat((self.visual_encoding, visual_encoding[:, :]), dim=0)
				self.label = np.concatenate((self.label, label[:]))
    
	def on_validation_begin(self):	
		self.curr_val_loss = 0.0
	
	def on_validation_end(self) -> None:
		# print('Train loss:', self.curr_train_loss, ' train steps : ', self.train_steps)
		# print('Val loss:', self.curr_val_loss, ' val steps : ', self.val_steps)
		# if self.best_val_loss is None or self.curr_val_loss < self.best_val_loss:
		# 	self.best_val_loss = self.curr_val_loss
   
		# if self.best_val_loss == self.curr_val_loss:
		if self.current_epoch % 10 == 0:
  
			idx = np.arange(self.visual_encoding.shape[0])

			# randomize numpy array
			np.random.shuffle(idx)

			self.logger.experiment.add_embedding(mat=self.visual_encoding[idx[:2000], :],
												 label_img=self.visual_patch[idx[:2000], :, :, :],
												 global_step=self.current_epoch,
												 metadata=self.label[idx[:2000]])

		self.curr_val_loss, self.val_steps = 0.0, 0
		self.curr_train_loss, self.train_steps = 0.0, 0

if __name__ == '__main__':
	# parse command line arguments
	parser = argparse.ArgumentParser(description='Dual Auto Encoder')
	parser.add_argument('--batch_size', type=int, default=256, metavar='N',
						help='input batch size for training (default: 512)')
	parser.add_argument('--epochs', type=int, default=1000, metavar='N',
						help='number of epochs to train (default: 1000)')
	parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
						help='learning rate (default: 3e-4)')
	parser.add_argument('--lambd', type=float, default=0.0051, help='lambd hyperparam')
	parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
						help='log directory (default: logs)')
	parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
						help='model directory (default: models)')
	parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
						help='number of GPUs to use (default: 1)')
	parser.add_argument('--latent_size', type=int, default=256, metavar='N',
						help='Size of the common latent space (default: 256)')
	parser.add_argument('--dataset_config_path', type=str, default='jackal_data/dataset_config_haresh_local.yaml')
	args = parser.parse_args()
	
	# check if the dataset config yaml file exists
	if not os.path.exists(args.dataset_config_path): raise FileNotFoundError(args.dataset_config_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dm = MyDataLoader(data_config_path=args.dataset_config_path, batch_size=args.batch_size)
	model = BarlowModel(lr=args.lr, latent_size=args.latent_size,
						inertial_shape=dm.inertial_shape, 
      					lambd=1./args.latent_size, per_device_batch_size=args.batch_size).to(device)

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
	model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
										  filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
										  monitor='val_loss', verbose=True)

	print("Training model...")
	trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
						 max_epochs=args.epochs,
						 callbacks=[model_checkpoint_cb],
						 log_every_n_steps=10,
						 distributed_backend='dp',
						 num_sanity_val_steps=0,
						 logger=True,
						#  benchmark = True,
						 )

	trainer.fit(model, dm)