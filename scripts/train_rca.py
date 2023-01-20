#!/usr/bin/env python3

"""code to train the Ride Comfort Aware navigation cost function network from the visual and inertial signals"""

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
    def __init__(self, latent_size=128, expt_save_path=None):
        super(NATURLCostModel, self).__init__()
        
        assert expt_save_path is not None, "Please provide the path to the experiment directory"
        
        self.visual_encoder = VisualEncoderModel(latent_size=latent_size)
        self.cost_net = CostNet(latent_size=latent_size)
        self.model = nn.Sequential(self.visual_encoder, self.cost_net)
        
        self.best_val_loss = 1000000.0
        self.cost_model_save_path = visual_encoder_weights.replace("visual_encoder", "cost_model")
                
    def forward(self, visual, inertial, leg, feet):
        self.model(visual.float())
    
    def softmax_with_temp(self, x, y, temp=1.0):
        x = x / temp
        y = y / temp
        x = torch.exp(x)
        y = torch.exp(y)
        return x / (x + y)
    
    def compute_rca_cost(self, inertial):
        # given the inertial signals, sum up the power spectral density of the inertial signals
        return torch.sum(inertial**2, dim=-1)
        
    def training_step(self, batch, batch_idx):
        patch1, patch2, inertial, leg, feet, _, _ = batch
        # sampleidx = sampleidx.cpu()

        # kmeanslabels = self.kmeanslabels[sampleidx]
        # preference_labels = [self.preferences[i] for i in kmeanslabels]
        
        cost1, rep1 = self.forward(patch1, inertial, leg, feet)
        cost2, rep2 = self.forward(patch2, inertial, leg, feet)
        
        rep1, rep2 = rep1.detach().cpu().detach().numpy(), rep2.detach().cpu().detach().numpy()
        
        labels1, labels2 = self.kmeansmodel.predict(rep1), self.kmeansmodel.predict(rep2)
        preference_labels1 = [self.preferences[i] for i in labels1]
        preference_labels2 = [self.preferences[i] for i in labels2]

        # compute the preference loss
        pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1) + 0.5*self.compute_preference_loss(cost2, preference_labels2)
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
        patch1, patch2, inertial, leg, feet, _, _ = batch
        # sampleidx = sampleidx.cpu()

        # kmeanslabels = self.kmeanslabels[sampleidx]
        # preference_labels = [self.preferences[i] for i in kmeanslabels]
        
        cost1, rep1 = self.forward(patch1, inertial, leg, feet)
        cost2, rep2 = self.forward(patch2, inertial, leg, feet)
        
        rep1, rep2 = rep1.detach().cpu().detach().numpy(), rep2.detach().cpu().detach().numpy()
        
        labels1, labels2 = self.kmeansmodel.predict(rep1), self.kmeansmodel.predict(rep2)
        preference_labels1 = [self.preferences[i] for i in labels1]
        preference_labels2 = [self.preferences[i] for i in labels2]

        # compute the preference loss
        pref_loss = 0.5*self.compute_preference_loss(cost1, preference_labels1) + 0.5*self.compute_preference_loss(cost2, preference_labels2)
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
        return torch.optim.AdamW(self.cost_net.parameters(), lr=3e-4, weight_decay=1e-5, amsgrad=True)

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--num_gpus','-g', type=int, default=1, metavar='N',
                        help='number of GPUs to use (default: 8)')
    parser.add_argument('--latent_size', type=int, default=512, metavar='N',
                        help='Size of the common latent space (default: 128)')
    parser.add_argument('--save', type=int, default=0, metavar='N',
                        help='Whether to save the k means model and encoders at the end of the run')
    parser.add_argument('--expt_save_path', type=str, default='/robodata/haresh92/spot-vrl/models/acc_0.99979/')
    parser.add_argument('--data_config_path', type=str, default='spot_data/data_config.yaml')
    args = parser.parse_args()
    
    model = NATURLCostModel(latent_size=128, visual_encoder_weights=os.path.join(args.expt_save_path, 'visual_encoder.pt'))
    dm = NATURLDataModule(data_config_path=args.data_config_path, batch_size=args.batch_size)
    
    # early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=1000)
    # # create model checkpoint only at end of training
    # model_checkpoint_cb = ModelCheckpoint(dirpath=args.expt_save_path,
    #                                       filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_',
    #                                       verbose=True,
    #                                       monitor='val_loss',
    #                                       save_weights_only=True)
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
    
    
    
    
        

    
        
        
        
        