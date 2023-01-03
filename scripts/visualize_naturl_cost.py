import rosbag
import numpy as np
import torch
import torch.nn as nn
from scripts.models import VisualEncoderModel
from termcolor import cprint
import cv2
from tqdm import tqdm
import albumentations as A

class CostVisualizer:
    def __init__(self, model_path):
        self.model_path = model_path
        
        visual_encoder = VisualEncoderModel(latent_size=128)
        cost_net = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1), nn.ReLU()
        )
        
        self.pred_model = nn.Sequential(visual_encoder, cost_net)
        # load weights of model
        model_state_dict = torch.load('/robodata/haresh92/spot-vrl/models/acc_0.99979/cost_model.pt')
        self.pred_model.load_state_dict(model_state_dict)
        self.pred_model.eval()
        
        self.conv = nn.LazyConv2d()
        
        cprint('Model loaded', 'green')
        
    def forward(self, bevimage: torch.Tensor, stride: int = 1):
        # pad bevimage with zeros of size 32 on all sides
        bevimage_padded = torch.zeros(bevimage.shape[0], bevimage.shape[1], bevimage.shape[2]+64, bevimage.shape[3]+64)
        bevimage_padded[:, :, 32:32+bevimage.shape[2], 32:32+bevimage.shape[3]] = bevimage
        
        # cost_pred is a grayscale image of size (batch_size, 1, patch_height, patch_width)
        cost_batch = []
        
        for b in range(bevimage_padded.shape[0]):
            cost_pred = []
            for i in range(0, bevimage_padded.shape[2], stride):
                cost_row = []
                for j in range(0, bevimage_padded.shape[3], stride):
                    # extract a patch of size 64x64 from the padded bevimage with stride 
                    # patch_64 = bevimage_padded[b, :, i*stride:i*stride+64, j*stride:j*stride+64].unsqueeze(0)
                    patch_64 = bevimage_padded[b, :, i:i+64, j:j+64].unsqueeze(0)
                    print('patch64 shape : ',patch_64.shape)
                    # pass the patch through the model
                    with torch.no_grad():
                        cost_row.append(self.pred_model(patch_64).item())

                cost_pred.append(cost_row)
            cost_batch.append(cost_pred)
        cost_batch = torch.tensor(np.asarray(cost_batch))
            
        # convert list of list of tensors to tensor
        return cost_batch
    
if __name__ == '__main__':
    costviz = CostVisualizer('/robodata/haresh92/spot-vrl/models/acc_0.99979/cost_model.pt')
    
    rosbag = rosbag.Bag('/robodata/eyang/data/2022-12-13/2022-12-13-07-39-25.bag')
    for topic, msg, t in rosbag.read_messages(topics=['/bev/single/compressed']):
        curr_bev_img = np.fromstring(msg.data, np.uint8)
        curr_bev_img = cv2.imdecode(curr_bev_img, cv2.IMREAD_COLOR)
        
        # resize the image by half
        # curr_bev_img = cv2.resize(curr_bev_img, (curr_bev_img.shape[1]//4, curr_bev_img.shape[0]//4))
        # resize to 640x480
        curr_bev_img = cv2.resize(curr_bev_img, (1024, 768))
        
        bevimage = curr_bev_img.copy()
        cv2.imwrite('/robodata/haresh92/spot-vrl/bevimage.png', bevimage)
        
        curr_bev_img = curr_bev_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # convert to tensor
        curr_bev_img = torch.from_numpy(curr_bev_img).unsqueeze(0)
        
        cprint('Predicting cost...', 'yellow')
        cost = costviz.forward(curr_bev_img, stride=64)
        cprint('Done', 'green')
        
        print('predicted cost shape: ', cost.shape)
        
        cost = cost.numpy()
        cost = (cost * 255.0).astype(np.uint8)
        print(cost.shape)
        cost = cost.transpose(1, 2, 0)
        cost = cv2.applyColorMap(cost, cv2.COLORMAP_JET)
        cost = cv2.resize(cost, (1024, 768)) 
        
        cv2.imwrite('/robodata/haresh92/spot-vrl/cost_prediction.png', cost)
        exit()
        