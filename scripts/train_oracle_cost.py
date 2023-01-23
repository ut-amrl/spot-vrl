#!/usr/bin/env python3

"""code to train the cost function network from the human preferences and the representations"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from scripts.train_naturl_representations import VisualEncoderModel, NATURLDataModule, ProprioceptionModel
from termcolor import cprint
import argparse
import numpy as np
import os
import pickle
from scripts.models import CostNet

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from datetime import datetime
import tensorboard

class NATURLCostModel(pl.LightningModule):
    def __init__(self, latent_size=128, visual_encoder_weights=None, temp=1.0):
        super(NATURLCostModel, self).__init__()
        assert visual_encoder_weights is not None, "visual_encoder_weights cannot be None"
        
        self.visual_encoder = VisualEncoderModel(latent_size=latent_size)
        # load the weights from the visual encoder
        cprint("Loading the weights from the visual encoder", "green")
        self.visual_encoder.load_state_dict(torch.load(visual_encoder_weights))
        # self.visual_encoder.eval()
        # cprint("Loaded the weights from the visual encoder", "green")
                
        self.cost_net = CostNet(latent_size=latent_size)
        self.temp = temp
        
        # hardcode the preferences here for now
        # self.preferences = [[2, 3, 5],[0,1], 6, 8, 4, 7]
        
        # self.preferences = {
        #     0: 5, # bush
        #     1: 0, # yellow_bricks
        #     2: 0, # pebble_sidewalk
        #     3: 3, # grass
        #     4: 1, # asphalt 
        #     5: 4, # marble_rocks
        #     6: 0, # cement_sidewalk
        #     7: 2, # mulch
        #     8: 0  # red_bricks
        # }
        
        self.preferences = {
            'asphalt': 1,
            'grass': 3,
            'mulch': 2,
            'pebble_pavement': 0,
            'yellow_brick': 0,
            'red_brick': 0,
            'concrete': 0,
            'marble_rock': 4,
            'bush': 5
        }
        
        self.best_val_loss = 1000000.0
        self.cost_model_save_path = visual_encoder_weights.replace("visual_encoder", "oracle_model")
        self.mseloss = torch.nn.MSELoss()
                
    def forward(self, visual):
        visual_encoding = self.visual_encoder(visual.float())
        return self.cost_net(visual_encoding)
    
    def softmax_with_temp(self, x, y, temp=1.0):
        x = torch.exp(x / temp)
        y = torch.exp(y / temp)
        return x / (x + y)
    
    def compute_preference_loss(self, cost, preference_labels, temp=1.0):
        # shuffle the batch and compute the cost per sample
        loss = 0.0
        for i in range(cost.shape[0]):
            # randomly select a sample from the batch
            j = torch.randint(0, cost.shape[0], (1,))[0]
            if preference_labels[i] < preference_labels[j]:
                loss += self.softmax_with_temp(cost[i], cost[j], temp=temp)
            elif preference_labels[i] > preference_labels[j]:
                loss += self.softmax_with_temp(cost[j], cost[i], temp=temp)
            else:
                loss += (cost[i] - cost[j])**2
        return loss / cost.shape[0]
    
    def compute_ranking_loss(self, cost, preference_labels):
        # convert list of preferences to a tensor
        preference_labels = torch.tensor(preference_labels).float().to(cost.device)
        return self.mseloss(cost.flatten(), preference_labels.flatten())
    
    def compute_sigmoid_ranking_loss(self, cost, preference_labels, total_classes=6):
        # first first MSE between the cost and the preference labels
        # then perform sigmoid on the cost and preference labels
        # for example, if the preference label is 2.0, then the sigmoid of MSE is
        # exp(x-2) / sum(exp(x-0), exp(x-1), exp(x-2), exp(x-3), exp(x-4), exp(x-5), exp(x-6), exp(x-7), exp(x-8))
        cost = cost.flatten()
        preference_labels = torch.tensor(preference_labels).float().to(cost.device)
        
        numer = torch.exp((cost - preference_labels) ** 2)
        
        denom = torch.zeros_like(cost)
        for i in range(cost.shape[0]):
            for j in range(total_classes):
                denom[i] += torch.exp((cost[i] - j)**2)
        loss = numer / denom
        return torch.sum(loss)
        
        
    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch
        
        cost1 = self.forward(patch1)
        cost2 = self.forward(patch2)
                
        preference_labels = [self.preferences[i] for i in label]

        # compute the preference loss
        # pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1, temp=self.temp) + \
        #     0.5*self.compute_preference_loss(cost2, preference_labels2, temp=self.temp)
        pref_loss = 0.5*self.compute_ranking_loss(cost1, preference_labels) + 0.5*self.compute_ranking_loss(cost2, preference_labels)
        
        # cost must be invariant to the viewpoint of the patch
        vpt_inv_loss = torch.mean((cost1 - cost2)**2)
        # penalty for the cost crossing 25.0
        penalty_loss = torch.mean(torch.relu(cost1 - 25.0)) + torch.mean(torch.relu(cost2 - 25.0))
        
        loss = pref_loss + 0.1 * vpt_inv_loss + penalty_loss
    
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_pref_loss", pref_loss, prog_bar=True, on_epoch=True)
        self.log("train_vpt_inv_loss", vpt_inv_loss, prog_bar=True, on_epoch=True)
        self.log("train_penalty_loss", penalty_loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, label, _ = batch
        
        cost1 = self.forward(patch1)
        cost2 = self.forward(patch2)
                
        preference_labels = [self.preferences[i] for i in label]

        # compute the preference loss
        # pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1, temp=self.temp) + \
        #     0.5*self.compute_preference_loss(cost2, preference_labels2, temp=self.temp)
        pref_loss = 0.5*self.compute_ranking_loss(cost1, preference_labels) + 0.5*self.compute_ranking_loss(cost2, preference_labels)
        
        # cost must be invariant to the viewpoint of the patch
        vpt_inv_loss = torch.mean((cost1 - cost2)**2)
        # penalty for the cost crossing 25.0
        penalty_loss = torch.mean(torch.relu(cost1 - 25.0)) + torch.mean(torch.relu(cost2 - 25.0))
        
        loss = pref_loss + 0.1 * vpt_inv_loss + penalty_loss
    
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_pref_loss", pref_loss, prog_bar=True, on_epoch=True)
        self.log("val_vpt_inv_loss", vpt_inv_loss, prog_bar=True, on_epoch=True)
        self.log("val_penalty_loss", penalty_loss, prog_bar=True, on_epoch=True)
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
            # wrap the visual encoder and the costnet in a single module
            # and save the state dict as a .pt file
            model = nn.Sequential(self.visual_encoder, self.cost_net)
            torch.save(model.state_dict(), self.cost_model_save_path)
            cprint('Saved the model with the best validation loss', 'green')
            # on last epoch, display the model save path
            if self.trainer.current_epoch == self.trainer.max_epochs - 1:
                cprint('The model was saved at {}'.format(self.cost_model_save_path), 'green', attrs=['bold'])
        cprint('the validation loss is {}'.format(val_loss), 'green')
    
    def configure_optimizers(self):
        # use only costnet parameters
        return torch.optim.AdamW(self.cost_net.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--num_gpus','-g', type=int, default=8, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--save', type=int, default=0, metavar='N',
                        help='Whether to save the k means model and encoders at the end of the run')
    parser.add_argument('--expt_save_path', '-e', type=str, default='/robodata/haresh92/spot-vrl/models/acc_0.98154_22-01-2023-05-13-46_')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    parser.add_argument('--temp', type=float, default=1.0)
    args = parser.parse_args()
    
    model = NATURLCostModel(latent_size=128, visual_encoder_weights=os.path.join(args.expt_save_path, 'visual_encoder.pt'), temp=args.temp)
    dm = NATURLDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="cost_training_logs/")
    
    print("Training the cost function model...")
    trainer = pl.Trainer(gpus=list(np.arange(args.num_gpus)),
                         max_epochs=args.epochs,
                        #  callbacks=[model_checkpoint_cb],
                         log_every_n_steps=10,
                         strategy='ddp',
                         num_sanity_val_steps=0,
                         sync_batchnorm=True,
                         logger=tb_logger
                         )

    # fit the model
    trainer.fit(model, dm)
    
    
    
    
        

    
        
        
        
        