#!/usr/bin/env python3
import torch.nn as nn
import torch
import torch.nn.functional as F

# create a pytorch model for the proprioception data
class ProprioceptionModel(nn.Module):
    def __init__(self, latent_size=64):
        super(ProprioceptionModel, self).__init__()
        # inertial encoder that takes an input of shape (batch_size, 1, 2800) and outputs a latent vector of shape (batch_size, latent_size)
        # should contain conv1d with increasing kernel_size and decreasing stride and increasing number of channels 
        # should contains batchnorm1d after each conv1d and relu after each batchnorm1d
        # final linear layer should have nn.Linear(1024, latent_size)
        
        # self.inertial_encoder = nn.Sequential( # input shape : (batch_size, 1, 2800)
        #     nn.Conv1d(1, 8, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(8), # output shape : (batch_size, 8, 1400)
        #     nn.Conv1d(8, 16, kernel_size=5, stride=3), nn.ReLU(), nn.BatchNorm1d(16), # output shape : (batch_size, 16, 466)
        #     nn.AvgPool1d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 233)
        #     nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(32), # output shape : (batch_size, 32, 116)
        #     nn.Conv1d(32, 64, kernel_size=5, stride=3), nn.ReLU(), nn.BatchNorm1d(64), # output shape : (batch_size, 64, 38)
        #     nn.AvgPool1d(kernel_size=3, stride=2), # output shape : (batch_size, 64, 18)
        #     nn.Conv1d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(128), # output shape : (batch_size, 128, 8)
        #     nn.AvgPool1d(kernel_size=5, stride=3), # output shape : (batch_size, 128, 2)
        #     nn.Flatten(),
        #     nn.Linear(256, 32)
        # )
        
        # self.leg_encoder = nn.Sequential( # input shape : (batch_size, 1, 1728)
        #     nn.Conv1d(1, 8, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(8), # output shape : (batch_size, 8, 864)
        #     nn.Conv1d(8, 16, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(16), # output shape : (batch_size, 16, 432)
        #     nn.AvgPool1d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 216)
        #     nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(32), # output shape : (batch_size, 32, 108)
        #     nn.Conv1d(32, 64, kernel_size=5, stride=3), nn.ReLU(), nn.BatchNorm1d(64), # output shape : (batch_size, 64, 36)
        #     nn.AvgPool1d(kernel_size=3, stride=2), # output shape : (batch_size, 64, 18)
        #     nn.Conv1d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(128), # output shape : (batch_size, 128, 8)
        #     nn.AvgPool1d(kernel_size=5, stride=3), # output shape : (batch_size, 128, 2)
        #     nn.Flatten(),
        #     nn.Linear(256, 32)
        # )
        
        # self.feet_encoder = nn.Sequential( # input shape : (batch_size, 1, 960)
        #     nn.Conv1d(1, 8, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(8), # output shape : (batch_size, 8, 480)
        #     nn.Conv1d(8, 16, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(16), # output shape : (batch_size, 16, 240)
        #     nn.AvgPool1d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 120)
        #     nn.Conv1d(16, 32, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(32), # output shape : (batch_size, 32, 60)
        #     nn.Conv1d(32, 64, kernel_size=5, stride=3), nn.ReLU(), nn.BatchNorm1d(64), # output shape : (batch_size, 64, 20)
        #     nn.AvgPool1d(kernel_size=3, stride=2), # output shape : (batch_size, 64, 10)
        #     nn.Conv1d(64, 128, kernel_size=3, stride=2), nn.ReLU(), nn.BatchNorm1d(128), # output shape : (batch_size, 128, 4)
        #     nn.AvgPool1d(kernel_size=2, stride=2), # output shape : (batch_size, 128, 2)
        #     nn.Flatten(),
        #     nn.Linear(256, 32)
        # )
        
        self.inertial_encoder = nn.Sequential( # input shape : (batch_size, 1, 1407)
            nn.Flatten(),
            nn.Linear(1407, 128, bias=False), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 32), nn.ReLU(),
        )
        
        self.leg_encoder = nn.Sequential( # input shape : (batch_size, 1, 900)
            nn.Flatten(),
            nn.Linear(900, 128, bias=False), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 32), nn.ReLU(),
        )
        
        self.feet_encoder = nn.Sequential( # input shape : (batch_size, 1, 500)
            nn.Flatten(),
            nn.Linear(500, 128, bias=False), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 32), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 32, latent_size), nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )
        
    def forward(self, inertial, leg, feet):
        inertial = self.inertial_encoder(inertial)
        leg = self.leg_encoder(leg)
        feet = self.feet_encoder(feet)
        
        features = self.fc(torch.cat([inertial, leg, feet], dim=1))
        
        # normalize the features
        features = F.normalize(features, dim=-1)
        
        return features
    

class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        self.encoder = nn.Sequential( # input shape : (batch_size, 3, 64, 64)
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 8, 32, 32)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 16, 16)
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 32, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 64, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(128), # output shape : (batch_size, 128, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 128, 2, 2)
            nn.Flatten(),
            nn.Linear(512, latent_size)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        
        x = F.normalize(x, dim=-1)
        return x