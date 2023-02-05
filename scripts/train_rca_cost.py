#!/usr/bin/env python3

"""code to train the RCA algorithm that assigns costs to patches of terrain based on the inertial signals"""

import torch
torch.multiprocessing.set_sharing_strategy('file_system') #https://github.com/pytorch/pytorch/issues/11201
import torch.nn as nn
import pytorch_lightning as pl
from scripts.train_naturl_representations import VisualEncoderModel, VisualEncoderEfficientModel
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from termcolor import cprint
import argparse
import numpy as np
import os, glob
import pickle, cv2
from scripts.models import CostNet, RCAModel
from scripts.utils import get_transforms, process_feet_data
import yaml
from tqdm import tqdm

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from datetime import datetime
import tensorboard
import albumentations as A
import matplotlib.pyplot as plt

# terrain_names = ['asphalt', 'bush', 'concrete', 'grass', 'marble_rock', 'mulch', 'pebble_pavement', 'red_brick', 'yellow_brick']
terrain_names = ['asphalt', 'bush', 'concrete', 'grass', 'marble_rock', 'pebble_pavement', 'red_brick', 'yellow_brick']

class RCATerrainDataset(Dataset):
    def __init__(self, pickle_files_root, cost_stats=None, img_augment=False):
        self.pickle_files_paths = glob.glob(pickle_files_root + '/*.pkl')
        self.label = pickle_files_root.split('/')[-2]
        self.cost_dict = cost_stats
        
        if img_augment:
            self.transforms = get_transforms()
        else:
            self.transforms = None
    
    def __len__(self):
        return len(self.pickle_files_paths)
    
    def __getitem__(self, idx):
        with open(self.pickle_files_paths[idx], 'rb') as f:
            data = pickle.load(f)
        patches, imu, feet, leg = data['patches'], data['imu'][:, :-4], data['feet'], data['leg']
        
        if self.cost_dict is None:
            # process the feet data to remove the mu and std values for non-contacting feet
            feet = process_feet_data(feet)
            
            # perform fft on the imu data
            imu = np.abs(np.fft.fft(imu, axis=0))
            leg = np.abs(np.fft.fft(leg, axis=0))
            feet = np.abs(np.fft.fft(feet, axis=0))
            
            # return only the positive frequencies
            imu = imu[:imu.shape[0]//2]
            leg = leg[:leg.shape[0]//2]
            feet = feet[:feet.shape[0]//2]
        
        else:
            # # compute cost for this patch and min-max normalize it
            # cost = 2* np.mean(imu[:, 0]) + np.mean(imu[:, 1]) + np.mean(imu[:, 5])
            # cost = (cost - self.cost_stats['min']) / (self.cost_stats['max'] - self.cost_stats['min'])
            
            # cost = 2 * np.mean(imu[:, 0]) + np.mean(imu[:, 1]) + np.mean(imu[:, 5])
            # cost = (cost - self.min_cost) / (self.max_cost - self.min_cost) # min-max normalize
            # cost = self.cost_dict[self.label] * cost
            
            cost = self.cost_dict[self.label]
            
        # sample 2 values between 0 and num_patches-1
        patch_1_idx, patch_2_idx = np.random.choice(len(patches), 2, replace=False)
        
        # # sample a number between 0 and (num_patches-1)/2
        # patch_1_idx = np.random.randint(0, len(patches)//2)
        # # sample a number between (num_patches-1)/2 and num_patches-1
        # patch_2_idx = np.random.randint(len(patches)//2, len(patches))
        
        patch1, patch2 = patches[patch_1_idx], patches[patch_2_idx]
        
        # convert BGR to RGB
        patch1, patch2 = cv2.cvtColor(patch1, cv2.COLOR_BGR2RGB), cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB)
        
        # apply the transforms
        if self.transforms is not None:
            patch1 = self.transforms(image=patch1)['image']
            patch2 = self.transforms(image=patch2)['image']
        
        # normalize the image patches
        patch1 = np.asarray(patch1, dtype=np.float32) / 255.0
        patch2 = np.asarray(patch2, dtype=np.float32) / 255.0
        
        # transpose
        patch1, patch2 = np.transpose(patch1, (2, 0, 1)), np.transpose(patch2, (2, 0, 1))
        
        if self.cost_dict is None:
            return np.asarray(patch1), np.asarray(patch2), imu, leg, feet, self.label, idx
        else:
            return np.asarray(patch1), np.asarray(patch2), cost, self.label

# create pytorch lightning data module
class RCADataModule(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=64, num_workers=2):
        super().__init__()
        
        # read the yaml file
        cprint('Reading the yaml file at : {}'.format(data_config_path), 'green')
        self.data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
        self.data_config_path = '/'.join(data_config_path.split('/')[:-1])

        self.batch_size, self.num_workers = batch_size, num_workers
        
        self.mean, self.std = {}, {}
        self.min, self.max = {}, {}
        
        # load the train and val datasets
        self.load()
        cprint('Train dataset size : {}'.format(len(self.train_dataset)), 'green')
        cprint('Val dataset size : {}'.format(len(self.val_dataset)), 'green')
        
        
    def load(self):
        
        # check if the data_statistics.pkl file exists
        if os.path.exists(self.data_config_path + '/rca_costs.pkl'):
            cprint('rca_costs.pkl file found!', 'green')
            rca_costs = pickle.load(open(self.data_config_path + '/rca_costs.pkl', 'rb'))

        else:
            # # find the mean and std of the train dataset
            # cprint('RCA_cost.pkl file not found!', 'yellow')
            # cprint('Finding the mean and std of the train dataset', 'green')
            # self.tmp_dataset = ConcatDataset([RCATerrainDataset(pickle_files_root) for pickle_files_root in self.data_config['train']])
            # self.tmp_dataloader = DataLoader(self.tmp_dataset, batch_size=512, num_workers=10, shuffle=False)
            # cprint('the length of the tmp_dataloader is : {}'.format(len(self.tmp_dataloader)), 'green')
            # # find the mean and std of the train dataset
            # imu_data, leg_data, feet_data, tlabels = [], [], [], []
            # for _, _, imu, leg, feet, tlabel, _ in tqdm(self.tmp_dataloader):
            #     imu_data.append(imu.cpu().numpy())
            #     leg_data.append(leg.cpu().numpy())
            #     feet_data.append(feet.cpu().numpy())
            #     tlabels.append(np.asarray(tlabel))
            # imu_data = np.concatenate(imu_data, axis=0)
            # leg_data = np.concatenate(leg_data, axis=0)
            # feet_data = np.concatenate(feet_data, axis=0)
            # tlabels = np.concatenate(tlabels, axis=0)
            
            # print('shapes : ', imu_data.shape, leg_data.shape, feet_data.shape, tlabels.shape)
            
            # # compute the costs of the train dataset. Note: only using IMU data
            # costs = np.zeros((imu_data.shape[0]))
            # for i in range(imu_data.shape[0]):
            #     costs[i] = 2 * np.mean(imu_data[i, :,  0]) + np.mean(imu_data[i, :,  1]) + np.mean(imu_data[i, :, 5])
            
            # costs = np.asarray(costs, dtype=np.float32)
            
            # # save costs and tlables in a pickle file
            # # pickle.dump({'costs' : costs, 'tlabels' : tlabels}, open(self.data_config_path + '/RCA_test.pkl', 'wb'))
            # # exit()
            
            # # normalize the cost between 0 and 1
            # cmin, cmax = np.min(costs), np.max(costs)
            # costs = (costs - cmin) / (cmax - cmin)
            
            # cost_dict = {}
            # for tcost, tlabel in zip(costs, tlabels):
            #     if tlabel not in cost_dict:
            #         cost_dict[tlabel] = []
            #     cost_dict[tlabel].append(tcost)
                
            # for key in cost_dict.keys():
            #     cost_dict[key] = np.asarray(cost_dict[key])
            #     cost_dict[key] = np.mean(cost_dict[key])
                
            
            # tsum = np.sum([np.exp(cost_dict[key]/0.01) for key in cost_dict.keys()])
            # cost_dict = {key: np.exp(cost_dict[key]/0.01)/tsum for key in cost_dict.keys()}
                
            # # print('cost dict : ', cost_dict)
            
            # # # compute the mean and std of the cost and plot it 
            # # plt.bar(cost_dict.keys(), [np.mean(cost_dict[key]) for key in cost_dict.keys()],
            # #         yerr=[np.std(cost_dict[key]) for key in cost_dict.keys()], align='center', alpha=0.5, ecolor='black', capsize=10)
            # # plt.xticks(rotation=45)
            # # plt.ylabel('Cost')
            # # plt.title('Costs for different labels')
            # # # prevent the labels from being cut off
            # # plt.tight_layout()
                
            # # # save the plot
            
            # # plt.savefig('rca_costs.png')
            # # exit()
            
            # # save the mean and std
            # cprint('Saving the cost statistics', 'green')
            # data_statistics = {'min': cmin, 'max': cmax, 'cost_dict': cost_dict}
            
            # pickle.dump(data_statistics, open(self.data_config_path + '/RCA_cost.pkl', 'wb'))
            raise Exception('rca_costs.pkl file not found!. Run the file scripts/plot_rca_cost.py to generate the file')
            
        # load the train data
        self.train_dataset = ConcatDataset([RCATerrainDataset(pickle_files_root, img_augment=True, cost_stats=rca_costs) for pickle_files_root in self.data_config['train']])
        self.val_dataset = ConcatDataset([RCATerrainDataset(pickle_files_root, cost_stats=rca_costs) for pickle_files_root in self.data_config['val']])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, 
                          drop_last= True if len(self.train_dataset) % self.batch_size != 0 else False,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, 
                          drop_last= False,
                          pin_memory=True)


class RCACostModel(pl.LightningModule):
    def __init__(self, latent_size=128, model_save_path=None, temp=1.0):
        super(RCACostModel, self).__init__()
        assert model_save_path is not None, "model_save_path cannot be None"

        # self.model = nn.Sequential(
        #     VisualEncoderModel(latent_size=latent_size),
        #     # VisualEncoderEfficientModel(latent_size=latent_size),
        #     CostNet(latent_size=latent_size),
        # )
        
        self.model = RCAModel(n_classes=len(terrain_names))
        
        self.best_val_loss = 1e8
        self.cost_model_save_path = model_save_path
        self.loss = torch.nn.SmoothL1Loss(reduction='mean')
        # self.loss = torch.nn.MSELoss(reduction='sum')
                
    def forward(self, visual):
        return self.model(visual)
        
    def training_step(self, batch, batch_idx):
        patch1, patch2, cost, label = batch
        # label is the terrain name. find its position in the terrain_names list
        label_idx = [terrain_names.index(l) for l in label]
        
        cost1, cost2 = self(patch1), self(patch2)
        
        # loss = 0.5 * (self.loss(cost1, cost.float()) + self.loss(cost2, cost.float()))
        loss = 0.5 * nn.CrossEntropyLoss()(cost1, torch.tensor(label_idx).to(cost1.device)) + \
            0.5 * nn.CrossEntropyLoss()(cost2, torch.tensor(label_idx).to(cost2.device))
    
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch1, patch2, cost, label = batch
        # label is the terrain name. find its position in the terrain_names list
        label_idx = [terrain_names.index(l) for l in label]
        
        cost1, cost2 = self(patch1), self(patch2)
        
        # loss = 0.5 * (self.loss(cost1, cost) + self.loss(cost2, cost))
        loss = 0.5 * nn.CrossEntropyLoss()(cost1, torch.tensor(label_idx).to(cost1.device)) + \
            0.5 * nn.CrossEntropyLoss()(cost2, torch.tensor(label_idx).to(cost2.device))
    
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    
    def on_validation_end(self):
        # run on the current GPU
        # get the validation loss and save the model if it is the best so far
        val_loss = self.trainer.callback_metrics["val_loss"]
        # aggregate the validation loss across all GPUs
        if self.trainer.world_size > 1:
            val_loss = torch.tensor(val_loss).cuda()
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            val_loss = val_loss / self.trainer.world_size
            val_loss = val_loss.cpu().numpy()
        
        # Save the cost function model : run this only on GPU 0
        if val_loss < self.best_val_loss and torch.cuda.current_device() == 0:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.cost_model_save_path)
            cprint('Saved the model with the best validation loss', 'green')
            # on last epoch, display the model save path
            if self.trainer.current_epoch == self.trainer.max_epochs - 1:
                cprint('The model was saved at {}'.format(self.cost_model_save_path), 'green', attrs=['bold'])
        cprint('the validation loss is {}'.format(val_loss), 'green')
    
    def configure_optimizers(self):
        # use only costnet parameters
        return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-7, amsgrad=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the RCA cost model')
    parser.add_argument('--batch_size', '-b', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--num_gpus','-g', type=int, default=8, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=128, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--model_save_path', '-e', type=str, default='/robodata/haresh92/spot-vrl/models/rca_model.pt')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    parser.add_argument('--temp', type=float, default=1.0)
    args = parser.parse_args()
    
    dm = RCADataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    model = RCACostModel(latent_size=args.latent_size, 
                            model_save_path=args.model_save_path, 
                            temp=args.temp)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="rca_training_logs/")
    
    print("Training the cost function model...")
    trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
                         max_epochs=args.epochs,
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         sync_batchnorm=True,
                         logger=tb_logger,
                         gradient_clip_val=100.0,
                         gradient_clip_algorithm='norm',
                         stochastic_weight_avg=True,
                         )

    # fit the model
    trainer.fit(model, dm)