"""
This script loads the CostNet and converts it to a TorchScript model.
"""
from scripts.models import CostNet, VisualEncoderModel
import torch
import torch.nn as nn
import sys

# pytorch version 
print('PyTorch version: ', torch.__version__)

pt_file_default_path = '/robodata/haresh92/spot-vrl/models/acc_0.91508_27-01-2023-16-15-12_/cost_model.pt'
# pt_file_default_path = '/robodata/haresh92/spot-vrl/models/acc_0.91508_27-01-2023-16-15-12_/cost_model_grass_eq.pt'
# pt_file_default_path = '/robodata/haresh92/spot-vrl/models/acc_0.98154_22-01-2023-05-13-46_/oracle_model_grass_eq.pt'

# pt_file_path is the first argument, else use the default path
pt_file_path = sys.argv[1] if len(sys.argv) > 1 else pt_file_default_path

jit_file_save_path =pt_file_path.replace('.pt', '.jit')

# load the model
visual_encoder = VisualEncoderModel(latent_size=128)
cost_net = CostNet(latent_size=128)
model = nn.Sequential(visual_encoder, cost_net)
model.load_state_dict(torch.load(pt_file_path))

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.float() / 255.0)
        
model = WrappedModel(model)

# convert to TorchScript
model = torch.jit.script(model)
# save the TorchScript model
torch.jit.save(model, jit_file_save_path)
print('Saved TorchScript model to: ', jit_file_save_path)