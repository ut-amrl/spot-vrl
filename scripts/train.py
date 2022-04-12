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
	def forward(self, input, size=256):
		return input.view(input.size(0), size, 2, 2)

class CustomDataset(Dataset):
	def __init__(self, pickle_file_path):
		self.pickle_file_path = pickle_file_path
		self.data = pickle.load(open(self.pickle_file_path, 'rb'))
		self.delay = 300

	def __len__(self):
		return len(self.data) - 300 - 75

	def __getitem__(self, idx):
		# skip the first 20 seconds and last 5 seconds
		idx = idx + self.delay

		patch = self.data[idx]['patches']
		# pick random from list
		patch_1 = patch[np.random.randint(0, len(patch))]
		patch_2 = patch[np.random.randint(0, len(patch))]
		patch_1 = patch_1.astype(np.float32) / 255.0
		patch_2 = patch_2.astype(np.float32) / 255.0



		# joint_positions = self.data[idx]['joint_positions'][-13:, :].flatten()
		# joint_velocities = self.data[idx]['joint_velocities'][-13:, :].flatten()
		# joint_accelerations = self.data[idx]['joint_accelerations'][-13:, :].flatten()
		# linear_velocity = self.data[idx]['linear_velocity'][-13:, [2]].flatten()
		# angular_velocity = self.data[idx]['angular_velocity'][-13:, [0, 1]].flatten()
		foot_depth_sensor = self.data[idx]['depth_info'][-13:, :].flatten()

		# imu = np.hstack((joint_positions, joint_velocities, joint_accelerations, linear_velocity, angular_velocity, foot_depth_sensor))
		imu = foot_depth_sensor
		return patch_1, patch_2, imu

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
		for _, _, i in tmp:
			i = i.numpy()
			tmp_list.append(i)
		tmp_list = np.asarray(tmp_list)
		self.mean = np.mean(tmp_list, axis=0)
		self.std = np.std(tmp_list, axis=0)
		self.inertial_shape = self.mean.shape[1]
		print('Data statistics have been found.')
		del tmp, tmp_list

		print('Train dataset size:', len(self.train_dataset))
		print('Val dataset size:', len(self.val_dataset))

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)


class DualAEModel(pl.LightningModule):
	def __init__(self, lr=3e-4, latent_size=64, mean=None, std=None, inertial_shape=None):
		super(DualAEModel, self).__init__()

		self.save_hyperparameters(
			'lr',
			'latent_size',
			'inertial_shape'
		)

		# self.visual_encoder = nn.Sequential(
		# 	nn.Conv2d(1, 32, kernel_size=4, stride=2),
		# 	nn.BatchNorm2d(32), nn.PReLU(),
		# 	nn.Conv2d(32, 64, kernel_size=4, stride=2),
		# 	nn.BatchNorm2d(64), nn.PReLU(),
		# 	nn.Conv2d(64, 128, kernel_size=4, stride=2),
		# 	nn.BatchNorm2d(128), nn.PReLU(),
		# 	nn.Conv2d(128, 256, kernel_size=4, stride=2),
		# 	Flatten(),
		# 	nn.Linear(1024, latent_size)
		# )
		#
		# self.visual_decoder = nn.Sequential(
		# 	nn.Linear(latent_size, 1024),
		# 	UnFlatten(),
		# 	nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
		# 	nn.BatchNorm2d(128), nn.PReLU(),
		# 	nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
		# 	nn.BatchNorm2d(64), nn.PReLU(),
		# 	nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.PReLU(),
		# 	nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
		# 	nn.Sigmoid(),
		# )

		self.visual_encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=2), # 63 x 63
			nn.BatchNorm2d(16), nn.PReLU(),
			nn.Conv2d(16, 32, kernel_size=3, stride=2), # 31 x 31
			nn.BatchNorm2d(32), nn.PReLU(),
			nn.Conv2d(32, 64, kernel_size=5, stride=2), # 14 x 14
			nn.BatchNorm2d(64), nn.PReLU(),
			nn.Conv2d(64, 128, kernel_size=5, stride=2),  # 5 x 5
			nn.BatchNorm2d(128), nn.PReLU(),
			nn.Conv2d(128, 256, kernel_size=3, stride=2),  # 2 x 2
			nn.PReLU(),
			Flatten(),
			nn.Linear(1024, latent_size)
		)

		self.visual_decoder = nn.Sequential(
			nn.Linear(latent_size, 1024),
			UnFlatten(), 											# 2 x 2
			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),  # 5 x 5
			nn.BatchNorm2d(128), nn.PReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), # 13 x 13
			nn.BatchNorm2d(64), nn.PReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2), #  31 x 31
			nn.BatchNorm2d(32), nn.PReLU(),
			nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2), #  63 x 63
			nn.PReLU(),
			nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2), # 128 x 128
			nn.Sigmoid(),
		)

		# self.inertial_encoder = nn.Sequential(
		# 	nn.Linear(inertial_shape, 512), nn.BatchNorm1d(512), nn.PReLU(),
		# 	nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(),
		# 	nn.Linear(256, 128), nn.PReLU(),
		# 	nn.Linear(128, latent_size)
		# )
		#
		# self.inertial_decoder = nn.Sequential(
		# 	nn.Linear(latent_size, 128), nn.BatchNorm1d(128), nn.PReLU(),
		# 	nn.Linear(128, 256), nn.BatchNorm1d(256), nn.PReLU(),
		# 	nn.Linear(256, 512), nn.PReLU(),
		# 	nn.Linear(512, inertial_shape)
		# )
		#
		#
		# self.projector = nn.Sequential(
		# 	nn.Linear(latent_size, 64), nn.ReLU(),
		# 	nn.Linear(64, 32)
		# )

		# self.inertial_encoder = Encoder(14, 41, 64)
		# self.inertial_decoder = Decoder(14, 64, 41)

		self.mse_loss = nn.MSELoss()
		self.lr = lr
		self.mean, self.std = torch.tensor(mean).float(), torch.tensor(std).float()
		self.cosine_sim_loss = nn.CosineSimilarity()

	def forward(self, visual_patch, imu_history):

		# visual Auto Encoder
		visual_encoding = self.visual_encoder(visual_patch)
		visual_patch_recon = self.visual_decoder(visual_encoding)
		# L2 normalize the embedding space
		# visual_encoding_projected = F.normalize(self.projector(visual_encoding), p=2, dim=1)

		# IMU Auto Encoder
		# inertial_encoding = self.inertial_encoder(imu_history)
		# imu_history_recon = self.inertial_decoder(inertial_encoding)
		# # L2 normalize the embedding space
		# inertial_encoding_projected = F.normalize(self.projector(inertial_encoding), p=2, dim=1)

		return visual_patch_recon, visual_encoding#, visual_encoding_projected, imu_history_recon, inertial_encoding, inertial_encoding_projected

	def training_step(self, batch, batch_idx):
		visual_patch_1 , visual_patch_2, imu_history = batch

		visual_patch_1 = visual_patch_1.unsqueeze(1).float()
		visual_patch_2 = visual_patch_2.unsqueeze(1).float()
		imu_history = imu_history.float()

		# normalize IMU info
		device = imu_history.device
		imu_history = (imu_history - self.mean.to(device)) / (self.std.to(device) + 1e-8)

		# print('visu shape : ', visual_patch.shape)
		# print('imu hist shape : ', imu_history.shape)

		# visual_patch_recon, visual_encoding, visual_encoding_projected, imu_history_recon, inertial_encoding, inertial_encoding_projected = self.forward(visual_patch, imu_history)
		visual_patch_recon, visual_encoding = self.forward(visual_patch_1, imu_history)

		visual_recon_loss = torch.mean((visual_patch_2 - visual_patch_recon) ** 2)
		# imu_history_recon_loss = torch.mean((imu_history - imu_history_recon) ** 2)
		# embedding_similarity_loss = torch.mean((visual_encoding_projected - inertial_encoding_projected) ** 2)
		rae_loss = (0.5 * visual_encoding.pow(2).sum(1)).mean() #+ (0.5 * inertial_encoding.pow(2).sum(1)).mean()

		loss = visual_recon_loss + 0.001 * rae_loss #+ imu_history_recon_loss + embedding_similarity_loss
		self.log('train_loss', loss, prog_bar=True, logger=True)
		self.log('train_visual_recon_loss', visual_recon_loss, prog_bar=False, logger=True)
		# self.log('train_imu_history_recon_loss', imu_history_recon_loss, prog_bar=False, logger=True)
		# self.log('train_embedding_similarity_loss', embedding_similarity_loss, prog_bar=False, logger=True)
		self.log('train_rae_loss', rae_loss, prog_bar=False, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		visual_patch_1, visual_patch_2, imu_history = batch

		visual_patch_1 = visual_patch_1.unsqueeze(1).float()
		visual_patch_2 = visual_patch_2.unsqueeze(1).float()
		imu_history = imu_history.float()

		# normalize IMU info
		device = imu_history.device
		imu_history = (imu_history - self.mean.to(device)) / (self.std.to(device) + 1e-8)

		# print('visual patch shape : ', visual_patch.shape)
		# print('imu hist shape : ', imu_history.shape)

		# visual_patch_recon, visual_encoding, visual_encoding_projected, imu_history_recon, inertial_encoding, inertial_encoding_projected = self.forward(visual_patch, imu_history)
		visual_patch_recon, visual_encoding = self.forward(visual_patch_1, imu_history)

		visual_recon_loss = torch.mean((visual_patch_2 - visual_patch_recon) ** 2)
		# imu_history_recon_loss = torch.mean((imu_history - imu_history_recon) ** 2)
		# embedding_similarity_loss = torch.mean((visual_encoding_projected - inertial_encoding_projected) ** 2)
		rae_loss = (0.5 * visual_encoding.pow(2).sum(1)).mean() #+ (0.5 * inertial_encoding.pow(2).sum(1)).mean()

		loss = visual_recon_loss + 0.001 * rae_loss #+ imu_history_recon_loss + embedding_similarity_loss
		self.log('val_loss', loss, prog_bar=True, logger=True)
		self.log('val_visual_recon_loss', visual_recon_loss, prog_bar=False, logger=True)
		# self.log('val_imu_history_recon_loss', imu_history_recon_loss, prog_bar=False, logger=True)
		# self.log('val_embedding_similarity_loss', embedding_similarity_loss, prog_bar=False, logger=True)
		self.log('val_rae_loss', rae_loss, prog_bar=False, logger=True)
		return loss

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		if self.current_epoch % 10 == 0:

			visual_patch_1, visual_patch_2, imu_history = batch

			visual_patch_1 = visual_patch_1.unsqueeze(1).float()
			visual_patch_2 = visual_patch_2.unsqueeze(1).float()
			imu_history = imu_history.float()
			# print('imu hist shape : ', imu_history.shape)

			# normalize IMU info
			device = imu_history.device
			imu_history = (imu_history - self.mean.to(device)) / (self.std.to(device) + 1e-8)

			# visual_patch_recon, visual_encoding, visual_encoding_projected, imu_history_recon, inertial_encoding, inertial_encoding_projected = self.forward(visual_patch, imu_history)
			visual_patch_recon, visual_encoding = self.forward(visual_patch_1, imu_history)

			# embeddings = torch.cat((visual_encoding, inertial_encoding), dim=0)
			# labels = ['V' for _ in range(visual_encoding.shape[0])] + ['I' for _ in range(inertial_encoding.shape[0])]

			visual_patch_tmp = visual_patch_2.float()[:20, :, :, :]
			visual_patch_recon_tmp = visual_patch_recon.float()[:20, :, :, :]

			visual_patch_tmp = torch.cat((visual_patch_tmp, visual_patch_tmp, visual_patch_tmp), dim=1)
			visual_patch_recon_tmp = torch.cat((visual_patch_recon_tmp, visual_patch_recon_tmp, visual_patch_recon_tmp), dim=1)


			grid_img_visual_patch = make_grid(torch.cat((visual_patch_tmp, visual_patch_recon_tmp), dim=2), nrow=20)

			if batch_idx == 0:
				self.visual_encoding = visual_encoding[:, :]
				self.visual_patch = visual_patch_2[:, :, :, :]
				self.grid_img_visual_patch = grid_img_visual_patch
			else:
				self.visual_encoding = torch.cat((self.visual_encoding, visual_encoding[:, :]), dim=0)
				self.visual_patch = torch.cat((self.visual_patch, visual_patch_2[:, :, :, :]), dim=0)


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
	parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
						help='number of GPUs to use (default: 1)')
	parser.add_argument('--latent_size', type=int, default=6, metavar='N',
						help='Size of the common latent space (default: 6)')
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dm = MyDataLoader(data_path=args.data_dir, batch_size=args.batch_size)
	model = DualAEModel(lr=args.lr, latent_size=args.latent_size, mean=dm.mean, std=dm.std, inertial_shape=dm.inertial_shape).to(device)

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








