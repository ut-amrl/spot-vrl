import glob

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from scipy import fftpack
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
import cv2
import random
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scripts.recurrent_model import Encoder, Decoder
from scripts.optimizer import LARS, CosineWarmupScheduler
from typing import List, Union, Tuple

label_dict = {
	'cement': 0,
	'smallrocks': 1,
	'grass': 2,
	'speedway': 3
}

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
		self.data = pickle.load(open(self.pickle_file_path, 'rb'))#[:20]
		self.delay = 300
		# self.delay = 0

		self.label = pickle_file_path.split('/')[-2]

	def __len__(self):
		return len(self.data) - 300 - 75

	def __getitem__(self, idx):
		# skip the first 20 seconds and last 5 seconds
		idx = idx + self.delay

		# randomly pick a patch from the list
		patch = random.sample(self.data[idx]['patches'], 1)[0]
		patch = patch.astype(np.float32) / 255.0
		patch = np.expand_dims(patch, axis=0)

		joint_positions = self.data[idx]['joint_positions'][-26:, :].flatten()
		joint_velocities = self.data[idx]['joint_velocities'][-26:, :].flatten()
		joint_accelerations = self.data[idx]['joint_accelerations'][-26:, :].flatten()
		linear_velocity = self.data[idx]['linear_velocity'][-26:, [2]].flatten()
		angular_velocity = self.data[idx]['angular_velocity'][-26:, [0, 1]].flatten()
		foot_depth_sensor = self.data[idx]['depth_info'][-26:, :].flatten()

		inertial_data = np.hstack((linear_velocity, angular_velocity, foot_depth_sensor, joint_positions, joint_velocities, joint_accelerations))
		inertial_data = np.expand_dims(inertial_data, axis=0)

		# randomize the inertial data
		# inertial_data = np.random.randn(1, inertial_data.shape[1])

		return patch, inertial_data, self.label

class MyDataLoader(pl.LightningDataModule):
	def __init__(self, data_path, batch_size=32):
		super(MyDataLoader, self).__init__()
		self.batch_size = batch_size
		self.data_path = data_path
		self.setup()

	def setup(self, stage=None):
		train_data_path = self.data_path + 'train/*/*.pkl'
		val_data_path = self.data_path + 'val/*/*.pkl'

		self.train_dataset = ConcatDataset([CustomDataset(file) for file in glob.glob(train_data_path)])
		self.val_dataset = ConcatDataset([CustomDataset(file) for file in glob.glob(val_data_path)])

		# find mean, std statistics of inertial data in the training set
		print('Finding mean and std statistics of inertial data in the training set...')
		tmp = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
		tmp_list = []
		for _, i, _ in tmp:
			i = i.numpy()
			tmp_list.append(i)
		tmp_list = np.asarray(tmp_list)
		self.mean = np.mean(tmp_list, axis=0)
		self.std = np.std(tmp_list, axis=0)
		self.inertial_shape = self.mean.shape[1] if len(self.mean.shape) > 1 else 1
		print('Inertial shape:', self.inertial_shape)
		print('Data statistics have been found.')
		del tmp, tmp_list

		print('Train dataset size:', len(self.train_dataset))
		print('Val dataset size:', len(self.val_dataset))

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)

class BarlowModel(pl.LightningModule):
	def __init__(self, lr=3e-4, latent_size=64,
				 mean=None, std=None, inertial_shape=None,
				 scale_loss:float=1.0/32, lambd:float=3.9e-6, weight_decay=1e-6,
				 per_device_batch_size=32, num_warmup_steps_or_ratio: Union[int, float] = 0.1):
		super(BarlowModel, self).__init__()

		self.save_hyperparameters(
			'lr',
			'latent_size',
			'inertial_shape',
			'scale_loss',
			'lambd',
			'weight_decay'
		)

		self.scale_loss = scale_loss
		self.lambd = lambd
		self.weight_decay = weight_decay
		self.lr = lr
		self.per_device_batch_size = per_device_batch_size
		self.num_warmup_steps_or_ratio = num_warmup_steps_or_ratio

		self.visual_encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=2, bias=False), # 63 x 63
			nn.BatchNorm2d(16), nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), # 31 x 31
			nn.BatchNorm2d(32), nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, bias=False), # 14 x 14
			nn.BatchNorm2d(64), nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),  # 5 x 5
			nn.BatchNorm2d(128), nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
			nn.ReLU(),
			Flatten(), # 1024 output
			nn.Linear(1024, 128)
		)

		self.inertial_encoder = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=3, stride=2, bias=False), nn.BatchNorm1d(16), nn.ReLU(), # 558
			nn.Conv1d(16, 32, kernel_size=5, stride=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(), # 185
			nn.Conv1d(32, 64, kernel_size=7, stride=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), # 60
			nn.Conv1d(64, 128, kernel_size=3, stride=2, bias=False), nn.BatchNorm1d(128), nn.ReLU(), # 30
			nn.Conv1d(128, 256, kernel_size=7, stride=3, bias=False), nn.ReLU(), # 8
			nn.Flatten(),
			nn.Linear(2048, 128)
		)

		self.projector = nn.Sequential(
			nn.Linear(128, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
			nn.Linear(1024, latent_size)
		)

		self.mean, self.std = torch.tensor(mean).float(), torch.tensor(std).float()

		# normalization layer for the representations z1 and z2
		self.bn = nn.BatchNorm1d(latent_size, affine=False)

	def forward(self, visual_patch, imu_history):

		# normalize inertial data
		device = imu_history.device
		imu_history = (imu_history - self.mean.to(device)) / (self.std.to(device) + 1e-8)

		z1 = self.projector(self.visual_encoder(visual_patch))
		z2 = self.projector(self.inertial_encoder(imu_history))

		# empirical cross-correlation matrix
		c = self.bn(z1).T @ self.bn(z2)

		# sum the cross-correlation matrix between all gpus
		c.div_(self.per_device_batch_size * self.trainer.num_processes)
		self.all_reduce(c)

		# use --scale-loss to multiply the loss by a constant factor
		# In order to match the code that was used to develop Barlow Twins,
		# the authors included an additional parameter, --scale-loss,
		# that multiplies the loss by a constant factor.
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
		off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
		loss = on_diag + self.lambd * off_diag #+ 1e-3*torch.mean((z1 - z2)**2)
		return loss

	def off_diagonal(self, x):
		# return a flattened view of the off-diagonal elements of a square matrix
		n, m = x.shape
		assert n == m
		return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

	def all_reduce(self, c):
		if torch.distributed.is_initialized():
			torch.distributed.all_reduce(c)

	def common_step(self, batch, batch_idx):
		visual, inertial, _ = batch
		loss = self(visual, inertial)
		return loss

	def training_step(self, batch, batch_idx):
		loss = self.common_step(batch, batch_idx)
		self.log('train_loss', loss, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		with torch.no_grad():
			loss = self.common_step(batch, batch_idx)
		self.log('val_loss', loss, prog_bar=True, logger=True)

	# def configure_optimizers(self):
	# 	optimizer = LARS(
	# 		self.parameters(),
	# 		lr=0,  # Initialize with a LR of 0
	# 		weight_decay=self.weight_decay,
	# 		weight_decay_filter=exclude_bias_and_norm,
	# 		lars_adaptation_filter=exclude_bias_and_norm
	# 	)
	#
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

	def configure_optimizers(self):
		return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if self.current_epoch % 10 == 0:

			visual_patch, imu_history, label = batch
			label = np.asarray(label)
			visual_patch = visual_patch.float()
			visual_encoding = self.visual_encoder(visual_patch)


			if batch_idx == 0:
				self.visual_encoding = visual_encoding[:, :]
				self.visual_patch = visual_patch[:, :, :, :]
				self.label = label[:]
			else:
				self.visual_patch = torch.cat((self.visual_patch, visual_patch[:, :, :, :]), dim=0)
				self.visual_encoding = torch.cat((self.visual_encoding, visual_encoding[:, :]), dim=0)
				self.label = np.concatenate((self.label, label[:]))

	def on_validation_end(self) -> None:
		if self.current_epoch % 10 == 0:
			idx = np.arange(self.visual_encoding.shape[0])

			# randomize numpy array
			np.random.shuffle(idx)

			self.logger.experiment.add_embedding(mat=self.visual_encoding[idx[:2000], :],
												 label_img=self.visual_patch[idx[:2000], :, :, :],
												 global_step=self.current_epoch,
												 metadata=self.label[idx[:2000]])

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
	parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
						help='input batch size for training (default: 32)')
	parser.add_argument('--epochs', type=int, default=1000, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
						help='learning rate (default: 0.001)')
	parser.add_argument('--data_dir', type=str, default='data/', metavar='N',
						help='data directory (default: data)')
	parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
						help='log directory (default: logs)')
	parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
						help='model directory (default: models)')
	parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
						help='number of GPUs to use (default: 1)')
	parser.add_argument('--latent_size', type=int, default=1024, metavar='N',
						help='Size of the common latent space (default: 6)')
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dm = MyDataLoader(data_path=args.data_dir, batch_size=args.batch_size)
	model = BarlowModel(lr=args.lr, latent_size=args.latent_size, mean=dm.mean, std=dm.std,
						inertial_shape=dm.inertial_shape, scale_loss=1.0, lambd=0.0051, per_device_batch_size=args.batch_size).to(device)

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
	model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
										  filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
										  monitor='val_loss', verbose=True)

	print("Training model...")
	trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
						 max_epochs=args.epochs,
						 callbacks=[model_checkpoint_cb],
						 log_every_n_steps=10,
						 precision=16,
						 distributed_backend='ddp',
						 num_sanity_val_steps=-1,
						 logger=True,
						 sync_batchnorm=True,
						 )

	trainer.fit(model, dm)








