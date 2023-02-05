import rosbag
import numpy as np
import torch
import torch.nn as nn
from scripts.models import VisualEncoderModel, CostNet, VisualEncoderEfficientModel, RCAModel, RCAModelWrapped
from termcolor import cprint
import cv2
from tqdm import tqdm
import albumentations as A
import torchvision
import torch.nn.functional as F
import argparse

class CostVisualizer:
    def __init__(self, model_path):
        self.model_path = model_path
        if 'rca' in self.model_path:
            self.model = RCAModel(8)
        else:
            visual_encoder = VisualEncoderModel(latent_size=128)
            # visual_encoder = VisualEncoderEfficientModel(latent_size=128)
            cost_net = CostNet(latent_size=128)
            self.model = nn.Sequential(visual_encoder, cost_net)
            
        # load weights of model
        model_state_dict = torch.load(self.model_path)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        cprint('Model loaded', 'green')
        
        if 'rca' in self.model_path:
            self.model = RCAModelWrapped(self.model, rca_costs_pkl_path='/robodata/haresh92/spot-vrl/spot_data/rca_costs.pkl')
        
    def forward(self, bevimage: torch.Tensor, stride: int = 1):
        """ 
        Args:
            bevimage: [C, H, W]
            stride: stride of the sliding window
        """
        patches = bevimage.unfold(0, 3, 3).unfold(1, 64, stride).unfold(2, 64, stride)
        patches = patches.contiguous().view(-1, 3, 64, 64)
        with torch.no_grad():
            # resize patches to 64x64
            # patches = F.interpolate(patches, size=(64, 64), mode='bilinear', align_corners=True)
            cost = self.model(patches)
        
        # find patches with sum of pixels == 0 and set their cost to 0
        idx = torch.sum(patches, dim=(1, 2, 3)) == 0
        cost[idx] = 0
    
        # costm = cost.view(704//stride, 1472//stride)
        costm = cost.view((704-64)//stride+1, (1472-64)//stride+1)
        
        cost = F.interpolate(costm.unsqueeze(0).unsqueeze(0), size=(704, 1472), mode='nearest')
        # cost = F.interpolate(costm.unsqueeze(0).unsqueeze(0), size=(704, 1472), mode='bilinear', align_corners=True)
        return cost
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/robodata/haresh92/spot-vrl/models/acc_0.99604_20-01-2023-08-13-17_/cost_model.pt')
    parser.add_argument('--bag_path', '-b', type=str, default='/robodata/eyang/data/2023-01-21/2023-01-21-15-41-04.bag')
    parser.add_argument('--output_path', '-o' ,type=str, default='cost_video.mp4', help='path to save the video, including the name w/ extension (.mp4)')
    args = parser.parse_args()
    
    if 'rca' in args.model_path:
        cprint('RCA model found. Using max_val = 1.0', 'green')
        max_val = 1.0
    elif 'oracle' in args.model_path:
        cprint('Oracle model found. Using max_val = 6.0', 'green')
        max_val = 6.0
    # elif 'naturl' in args.model_path:
    #     cprint('NATURL model found. Using max_val = 6.0', 'green')
    #     max_val = 6.0
    else:
        # raise ValueError('Unknown model')
        max_val = 6.0
    
    costviz = CostVisualizer(args.model_path)
    rosbag = rosbag.Bag(args.bag_path)
    
    # find number of frames in the rosbag
    numframes = rosbag.get_message_count(topic_filters=['/bev/single/compressed'])
    
    stacked_img_list = []
    
    # save all the stacked images in a list and save in disk as a MPEG-4 video
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (1472*2, 704))

    # use tqdm and rosbag to iterate over the frames
    for topic, msg, t in tqdm(rosbag.read_messages(topics=['/bev/single/compressed']), total=numframes):
        curr_bev_img = np.fromstring(msg.data, np.uint8)
        curr_bev_img = cv2.imdecode(curr_bev_img, cv2.IMREAD_COLOR)
        curr_bev_img = cv2.cvtColor(curr_bev_img, cv2.COLOR_BGR2RGB) # img size is (749, 1476, 3)
        # remove the bottom and right part of the image to get a size of (704, 1472, 3)
        curr_bev_img = curr_bev_img[:704, :1472, :] #(64*11, 64*23, 3)
        
        bevimage = curr_bev_img.copy()
        
        curr_bev_img = curr_bev_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # convert to tensor
        curr_bev_img = torch.from_numpy(curr_bev_img)
        
        cost = costviz.forward(curr_bev_img, stride=64).squeeze(0).squeeze(0)
        
        cost = cost.numpy()
        cost = (cost * 255.0 / max_val).astype(np.uint8)
        cost = cv2.cvtColor(cost, cv2.COLOR_GRAY2RGB)
        cost = cv2.resize(cost, (1472, 704))
        
        stacked_img = cv2.cvtColor(np.hstack((bevimage, cost)), cv2.COLOR_RGB2BGR)
        out.write(stacked_img)
            
        # if ctrl+c is pressed, break the loop, save the video and exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            break

    # out.release()
        
        
        