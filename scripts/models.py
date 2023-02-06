#!/usr/bin/env python3
import torch.nn as nn
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
from termcolor import cprint

# create a pytorch model for the proprioception data
class ProprioceptionModel(nn.Module):
    def __init__(self, latent_size=64, p=0.05):
        super(ProprioceptionModel, self).__init__()
        
        self.inertial_encoder = nn.Sequential( # input shape : (batch_size, 1, 603)
            nn.Flatten(),
            nn.Linear(201*3, 128, bias=False), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, latent_size//2), nn.ReLU(),
        )
        
        self.leg_encoder = nn.Sequential( # input shape : (batch_size, 1, 900)
            nn.Flatten(),
            nn.Linear(900, 128, bias=False), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, latent_size//2), nn.ReLU(),
        )
        
        self.feet_encoder = nn.Sequential( # input shape : (batch_size, 1, 500)
            nn.Flatten(),
            nn.Linear(500, 128, bias=False), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, latent_size//2), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3 * latent_size//2, latent_size), nn.ReLU(),
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
    
class RCAModel(nn.Module):
    def __init__(self, n_classes=6):
        super(RCAModel, self).__init__()
        self.model = nn.Sequential( # input shape : (batch_size, 3, 64, 64)
            nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8 x 32 x 32
            nn.Conv2d(8, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 16 x 16
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x 8 x 8
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 x 4 x 4
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 x 2 x 2
            nn.Flatten(),
            nn.Linear(128*2*2, n_classes),
        )
    
    def forward(self, x):
        return self.model(x)
    
class RCAModelWrapped(nn.Module):
    def __init__(self, model, rca_costs_pkl_path=None):
        super(RCAModelWrapped, self).__init__()
        self.model = model
        assert rca_costs_pkl_path is not None, "rca_costs_pkl_path is None"
        cprint("Loading WrappedModel for RCA", "green")
        self.rca_costs = pickle.load(open(rca_costs_pkl_path, "rb"))
        self.terrain_classes = list(self.rca_costs.keys())
    def forward(self, x):
        x = self.model(x)
        # get the class
        class_ = torch.argmax(x, dim=1)
        # get the cost
        return torch.tensor([self.rca_costs[self.terrain_classes[tc]] for tc in class_])
        

class CostNet(nn.Module):
    def __init__(self, latent_size=64):
        super(CostNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size, latent_size//2), nn.BatchNorm1d(latent_size//2), nn.ReLU(),
            nn.Linear(latent_size//2, 1), nn.Sigmoid(), #nn.ReLU(), #nn.Softplus(), 
        )
        
    def forward(self, x):
        return self.fc(x)
    
class VisualEncoderEfficientModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderEfficientModel, self).__init__()
        
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        del self.model._fc
        self.model._fc = nn.Linear(1280, latent_size)
        
    def forward(self, x):
        # image is between 0 and 1, normalize it to -1 and 1
        # x = x * 2 - 1
        x = self.model(x)
        x = F.normalize(x, dim=-1)
        return x
    
# class VisualEncoderModel(nn.Module):
#     def __init__(self, latent_size=64, n_heads=4) -> None:
#         super(VisualEncoderModel, self).__init__()
#         # build a simple vision transformer
#         width, height = 64, 64
        
#         patch_height, patch_width = 8, 8
#         num_patches = (width // patch_width) * (height // patch_height)
#         patch_dim = 3 * patch_height * patch_width
        
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, latent_size)
#         )
        
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, latent_size))
#         self.class_token = nn.Parameter(torch.randn(1, 1, latent_size))
#         self.embedding_dropout = nn.Dropout(0.1)
#         encoder_layer = nn.TransformerEncoderLayer(d_model = latent_size, nhead = n_heads, dim_feedforward=128, activation='gelu')
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

#     def forward(self, x):
#         # image is of shape (batch_size, 3, 64, 64)
#         x = self.to_patch_embedding(x)
#         batch_size, n, _ = x.shape
#         class_tokens = repeat(self.class_token, '() n d -> b n d', b = batch_size)
#         x = torch.cat((class_tokens, x), dim=1)
#         x = x + self.pos_embedding[:, :(n + 1)]
#         x = self.embedding_dropout(x)
#         x = rearrange(x, 'b n d -> n b d')
#         x = self.transformer(x)
#         return x[0]
        
# class VisualEncoderModel(nn.Module):
#     def __init__(self, latent_size=64) -> None:
#         super(VisualEncoderModel, self).__init__()
        
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 64, 64),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 8, 32, 32),
#         )
        
#         self.skip1 = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 64, 64),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 8, 32, 32),
#         )
        
#         self.block2 = nn.Sequential(
#             nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 32, 32),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 16, 16),
#         )
        
#         self.skip2 = nn.Sequential(
#             nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 32, 32),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 16, 16),
#         )
        
#         self.block3 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 16, 16),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 32, 8, 8),
#         )
        
#         self.skip3 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 16, 16),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 32, 8, 8),
#         )
        
#         self.block4 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 8, 8),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 64, 4, 4),
#         )
        
#         self.skip4 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 8, 8),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 64, 4, 4),
#         )
        
#         self.block5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(128), # output shape : (batch_size, 128, 4, 4),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 128, 2, 2),
#         )
        
#         self.skip5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(128), # output shape : (batch_size, 128, 4, 4),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 128, 2, 2),
#         )
        
#         self.block6 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(256), # output shape : (batch_size, 256, 2, 2),
#             nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 256, 1, 1),
#         )
        
#         self.fc = nn.Sequential(
#             nn.Linear(256, latent_size), nn.ReLU(),
#         )
        
#     def forward(self, x):
#         x = self.block1(x) + self.skip1(x)
#         x = self.block2(x) + self.skip2(x)
#         x = self.block3(x) + self.skip3(x)
#         x = self.block4(x) + self.skip4(x)
#         x = self.block5(x) + self.skip5(x)
#         x = self.block6(x) 
        
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
        
#         x = F.normalize(x, dim=-1)
        
#         return x 
        
    
# class VisualEncoderModel(nn.Module):
#     def __init__(self, latent_size=64):
#         super(VisualEncoderModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(8), nn.PReLU(), # output shape : (batch_size, 8, 31, 31),
#             nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(16), nn.PReLU(), # output shape : (batch_size, 16, 15, 15),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(32), nn.PReLU(), # output shape : (batch_size, 32, 7, 7),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.PReLU(), # output shape : (batch_size, 64, 3, 3),
#             nn.AvgPool2d(kernel_size=3), # output shape : (batch_size, 64, 1, 1),
#             nn.Flatten(),
#             nn.Linear(64, latent_size), nn.ReLU(),
#         )
    
#     def forward(self, x):
#         # return F.normalize(self.model(x), dim=-1)
#         return self.model(x)
    
class VisualEncoderModel(nn.Module):
    def __init__(self, latent_size=64):
        super(VisualEncoderModel, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 8, 32, 32),
        )
        
        self.skipblock = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 32, 32),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(8), # output shape : (batch_size, 8, 32, 32),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 16, 16, 16),
        )
        
        self.skipblock2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 16, 16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(16), # output shape : (batch_size, 16, 16, 16),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 16, 16),
            nn.AvgPool2d(kernel_size=2, stride=2), # output shape : (batch_size, 32, 8, 8),
        )
        
        self.skipblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 8, 8),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(32), # output shape : (batch_size, 32, 8, 8),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 2, 2),
        )
        
        self.skipblock4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(), nn.BatchNorm2d(64), # output shape : (batch_size, 64, 2, 2),
        )
        
        self.fc = nn.Linear(256, latent_size)
        
    
    def forward(self, x):
        x = self.block1(x)
        x = self.skipblock(x) + x
        x = self.block2(x)
        x = self.skipblock2(x) + x
        x = self.block3(x)
        x = self.skipblock3(x) + x
        x = self.block4(x)
        x = self.skipblock4(x) + x
        x = x.view(x.size(0), -1) # flattened to (batch_size, 256)
        
        x = self.fc(x)
        
        # normalize
        x = F.normalize(x, dim=-1)
        
        return x


# class MobileNetBlock(nn.Module):
#     def __init__(self, input_channel, t, c, s=1):
#         super(MobileNetBlock, self).__init__()
#         self.stride = s

#         self.skip_connection = nn.Sequential(
#             nn.Conv2d(in_channels=input_channel, out_channels=c, kernel_size=1, stride=self.stride,
#                       bias=False),
#             nn.BatchNorm2d(c),
#         )

#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=input_channel, out_channels= t * input_channel, kernel_size=1, stride=1,
#                       bias=False),
#             nn.BatchNorm2d(t * input_channel),
#             nn.ReLU6(),
#             nn.Conv2d(in_channels=t * input_channel, out_channels=t * input_channel, kernel_size=3, stride=self.stride,
#                       bias=False, padding=1, groups=t * input_channel),
#             nn.BatchNorm2d(t * input_channel),
#             nn.ReLU6(),
#             nn.Conv2d(in_channels=t * input_channel, out_channels = c, kernel_size=1, stride=1,
#                       bias=False),
#             nn.BatchNorm2d(c),
#         )

#     def forward(self, x):
#         skip_connection = self.skip_connection(x)

#         x = self.block(x)
#         if self.stride == 1:
#             x = skip_connection + x

#         return x

# class VisualEncoderModel(nn.Module):
#     def __init__(self, latent_size=64):
#         super(VisualEncoderModel, self).__init__()

#         self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
#         self.conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

#         self.block1 = MobileNetBlock(32, 1, 16, 1)

#         self.block2 = MobileNetBlock(16, 6, 24, 1)

#         self.block2_1 = MobileNetBlock(24, 6, 24, 1)

#         self.block3 = MobileNetBlock(24, 6, 32, 2)

#         self.block3_1 = MobileNetBlock(32, 6, 32, 1)

#         self.block4 = MobileNetBlock(32, 6, 64, 2)

#         self.block4_1 = MobileNetBlock(64, 6, 64, 1)
        
#         self.block5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)

#         self.avgpool = nn.AvgPool2d(kernel_size=5)

#         self.fc = nn.Linear(256, latent_size)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.maxpool(x)

#         x = self.conv_1(x)
#         x = self.maxpool(x)

#         x = self.block1(x)
#         #print(b1.shape)

#         x = self.block2(x)

#         x = self.block2_1(x)
#         #print(b2.shape)

#         x = self.block3(x)
#         x = self.block3_1(x)
#         # x = self.block3_1(x)
#         #print(b3.shape)

#         x = self.block4(x)
#         x = self.block4_1(x)
#         x = self.block4_1(x)
#         # x = self.block4_1(x)
#         #print(b4.shape)
        
#         x = self.block5(x)
        
#         x = self.avgpool(x)

#         x = x.view(x.size(0), -1) # flattened to (batch_size, 576)

#         x = self.fc(x)
#         #print(x.shape)

#         # normalize
#         x = F.normalize(x, dim=-1)
        
#         return x