import glob

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
import cv2
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from scripts.recurrent_model import Encoder, Decoder

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
	def forward(self, input, size=1024):
		return input.view(input.size(0), size, 1, 1)

class CustomDataset(Dataset):
	def __init__(self, pickle_file_path):
		self.pickle_file_path = pickle_file_path
		self.data = pickle.load(open(self.pickle_file_path, 'rb'))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		patch = self.data[idx]['patches']
		# pick random from list
		patch = patch[np.random.randint(0, len(patch))]
		patch = patch / 255.0

		joints = self.data[idx]['inertial'][-13:, 1:37]
		linear_vel = self.data[idx]['inertial'][-13:, 37:37+3]
		angular_vel = self.data[idx]['inertial'][-13:, 40:40+3]
		foot = self.data[idx]['inertial'][-13:, 43:43+4]

		imu = np.hstack((joints.flatten(), linear_vel[:, [2]].flatten(), angular_vel[:, [0, 1]].flatten(), foot.flatten()))

		return patch, imu

class MyDataLoader(pl.LightningDataModule):
	def __init__(self, data_path, batch_size=32, smaller_data=False):
		super(MyDataLoader, self).__init__()
		self.batch_size = batch_size
		self.data_path = data_path
		self.smaller_data = smaller_data

	def setup(self, stage=None):
		train_data_path = self.data_path + 'train/*/*.pkl'
		val_data_path = self.data_path + 'val/*/*.pkl'
		if self.smaller_data:
			train_data_path = self.data_path + 'train/*/*_short.pkl'
			val_data_path = self.data_path + 'val/*/*_short.pkl'

		self.train_dataset = ConcatDataset([CustomDataset(file) for file in glob.glob(train_data_path)])
		self.val_dataset = ConcatDataset([CustomDataset(file) for file in glob.glob(val_data_path)])

		print('Train dataset size:', len(self.train_dataset))
		print('Val dataset size:', len(self.val_dataset))

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


class DualAEModel(pl.LightningModule):
	def __init__(self, lr=3e-4):
		super(DualAEModel, self).__init__()
		self.visual_encoder = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=4, stride=2),
			nn.BatchNorm2d(32), nn.PReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64), nn.PReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.BatchNorm2d(128), nn.PReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			Flatten(),
			nn.Linear(1024, 64)
		)

		self.visual_decoder = nn.Sequential(
			nn.Linear(64, 1024),
			UnFlatten(),
			nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
			nn.BatchNorm2d(128), nn.PReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
			nn.BatchNorm2d(64), nn.PReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.PReLU(),
			nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
			nn.Sigmoid(),
		)

		self.inertial_encoder = nn.Sequential(
			nn.Linear(559, 512), nn.BatchNorm1d(512), nn.PReLU(),
			nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(),
			nn.Linear(256, 128), nn.PReLU(),
			nn.Linear(128, 64)
		)

		self.inertial_decoder = nn.Sequential(
			nn.Linear(64, 128), nn.BatchNorm1d(128), nn.PReLU(),
			nn.Linear(128, 256), nn.BatchNorm1d(256), nn.PReLU(),
			nn.Linear(256, 512), nn.PReLU(),
			nn.Linear(512, 559)
		)

		# self.inertial_encoder = Encoder(14, 41, 64)
		# self.inertial_decoder = Decoder(14, 64, 41)

		self.mse_loss = nn.MSELoss()
		self.lr = lr

	def forward(self, visual_patch, imu_history):

		# visual Auto Encoder
		visual_encoding = self.visual_encoder(visual_patch)
		# L2 normalize the embedding space
		# visual_encoding = F.normalize(visual_encoding, p=2, dim=1)
		visual_patch_recon = self.visual_decoder(visual_encoding)

		# IMU Auto Encoder
		inertial_encoding = self.inertial_encoder(imu_history)
		# L2 normalize the embedding space
		# inertial_encoding = F.normalize(inertial_encoding, p=2, dim=1)
		imu_history_recon = self.inertial_decoder(inertial_encoding)

		return visual_patch_recon, visual_encoding, imu_history_recon, inertial_encoding

	def training_step(self, batch, batch_idx):
		visual_patch , imu_history = batch

		visual_patch = visual_patch.unsqueeze(1).float()
		imu_history = imu_history.float()

		# print('visu shape : ', visual_patch.shape)
		# print('imu hist shape : ', imu_history.shape)

		visual_patch_recon, visual_encoding, imu_history_recon, inertial_encoding = self.forward(visual_patch,
																								 imu_history)

		visual_recon_loss = torch.mean((visual_patch - visual_patch_recon) ** 2)
		imu_history_recon_loss = torch.mean((imu_history - imu_history_recon) ** 2)
		embedding_similarity_loss = torch.mean((visual_encoding - inertial_encoding) ** 2)
		rae_loss = (0.5 * visual_encoding.pow(2).sum(1)).mean() + (0.5 * inertial_encoding.pow(2).sum(1)).mean()

		loss = visual_recon_loss + imu_history_recon_loss + embedding_similarity_loss + rae_loss
		self.log('train_loss', loss, prog_bar=True, logger=True)
		self.log('train_visual_recon_loss', visual_recon_loss, prog_bar=False, logger=True)
		self.log('train_imu_history_recon_loss', imu_history_recon_loss, prog_bar=False, logger=True)
		self.log('train_embedding_similarity_loss', embedding_similarity_loss, prog_bar=False, logger=True)
		self.log('train_rae_loss', rae_loss, prog_bar=False, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		visual_patch , imu_history = batch

		visual_patch = visual_patch.unsqueeze(1).float()
		imu_history = imu_history.float()

		# print('visual patch shape : ', visual_patch.shape)
		# print('imu hist shape : ', imu_history.shape)

		visual_patch_recon, visual_encoding, imu_history_recon, inertial_encoding = self.forward(visual_patch,
																								 imu_history)

		visual_recon_loss = torch.mean((visual_patch - visual_patch_recon) ** 2)
		imu_history_recon_loss = torch.mean((imu_history - imu_history_recon) ** 2)
		embedding_similarity_loss = torch.mean((visual_encoding - inertial_encoding) ** 2)
		rae_loss = (0.5 * visual_encoding.pow(2).sum(1)).mean() + (0.5 * inertial_encoding.pow(2).sum(1)).mean()

		loss = visual_recon_loss + imu_history_recon_loss + embedding_similarity_loss + rae_loss
		self.log('val_loss', loss, prog_bar=True, logger=True)
		self.log('val_visual_recon_loss', visual_recon_loss, prog_bar=False, logger=True)
		self.log('val_imu_history_recon_loss', imu_history_recon_loss, prog_bar=False, logger=True)
		self.log('val_embedding_similarity_loss', embedding_similarity_loss, prog_bar=False, logger=True)
		self.log('val_rae_loss', rae_loss, prog_bar=False, logger=True)
		return loss

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if self.current_epoch % 10 == 0:

			visual_patch, imu_history = batch
			visual_patch = visual_patch.unsqueeze(1).float()
			imu_history = imu_history.float()
			# print('imu hist shape : ', imu_history.shape)

			visual_patch_recon, visual_encoding, imu_history_recon, inertial_encoding = self.forward(visual_patch,
																									 imu_history)
			# embeddings = torch.cat((visual_encoding, inertial_encoding), dim=0)
			# labels = ['V' for _ in range(visual_encoding.shape[0])] + ['I' for _ in range(inertial_encoding.shape[0])]

			visual_patch_tmp = visual_patch.float()[:20, :, :, :]
			visual_patch_recon_tmp = visual_patch_recon.float()[:20, :, :, :]

			visual_patch_tmp = torch.cat((visual_patch_tmp, visual_patch_tmp, visual_patch_tmp), dim=1)
			visual_patch_recon_tmp = torch.cat((visual_patch_recon_tmp, visual_patch_recon_tmp, visual_patch_recon_tmp), dim=1)


			grid_img_visual_patch = make_grid(torch.cat((visual_patch_tmp, visual_patch_recon_tmp), dim=2), nrow=20)

			if batch_idx == 0:
				self.visual_encoding = visual_encoding[:20, :]
				self.visual_patch = visual_patch[:20, :, :, :]
				self.grid_img_visual_patch = grid_img_visual_patch
			else:
				self.visual_encoding = torch.cat((self.visual_encoding, visual_encoding[:20, :]), dim=0)
				self.visual_patch = torch.cat((self.visual_patch, visual_patch[:20, :, :, :]), dim=0)


	def on_validation_end(self) -> None:
		if self.current_epoch % 10 == 0:
			self.logger.experiment.add_embedding(mat=self.visual_encoding, label_img=self.visual_patch, global_step=self.current_epoch)
			self.logger.experiment.add_image('visual_recons', self.grid_img_visual_patch, self.current_epoch)


if __name__ == '__main__':
	# parse command line arguments
	parser = argparse.ArgumentParser(description='Dual Auto Encoder')
	parser.add_argument('--batch_size', type=int, default=128, metavar='N',
						help='input batch size for training (default: 32)')
	parser.add_argument('--epochs', type=int, default=1000, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.001)')
	parser.add_argument('--data_dir', type=str, default='data/', metavar='N',
						help='data directory (default: data)')
	parser.add_argument('--log_dir', type=str, default='logs/', metavar='N',
						help='log directory (default: logs)')
	parser.add_argument('--model_dir', type=str, default='models/', metavar='N',
						help='model directory (default: models)')
	parser.add_argument('--smaller_data', action='store_true', default=False)
	parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
						help='number of GPUs to use (default: 1)')
	args = parser.parse_args()

	if args.smaller_data: print('Using smaller dataset..')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dm = MyDataLoader(data_path=args.data_dir, batch_size=args.batch_size, smaller_data=args.smaller_data)
	model = DualAEModel(lr=args.lr).to(device)

	early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)
	model_checkpoint_cb = ModelCheckpoint(dirpath='models/',
										  filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
										  monitor='val_loss', verbose=True)

	print("Training model...")
	trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
						 max_epochs=args.epochs,
						 callbacks=[early_stopping_cb, model_checkpoint_cb],
						 log_every_n_steps=10,
						 distributed_backend='ddp',
						 num_sanity_val_steps=-1,
						 stochastic_weight_avg=True,
						 gradient_clip_val=1.0,
						 logger=True,
						 )

	trainer.fit(model, dm)








