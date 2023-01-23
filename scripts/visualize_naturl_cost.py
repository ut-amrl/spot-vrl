import rosbag
import numpy as np
import torch
import torch.nn as nn
from scripts.models import VisualEncoderModel, CostNet
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
        
        visual_encoder = VisualEncoderModel(latent_size=128)
        cost_net = CostNet(latent_size=128)

        self.model = nn.Sequential(visual_encoder, cost_net)
        # load weights of model
        model_state_dict = torch.load(self.model_path)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        cprint('Model loaded', 'green')
        
    def forward(self, bevimage: torch.Tensor, stride: int = 1):
        """
        Args:
            bevimage: [C, H, W]
            stride: stride of the sliding window
        """
        patches = bevimage.unfold(0, 3, 3).unfold(1, 64, stride).unfold(2, 64, stride)
        patches = patches.contiguous().view(-1, 3, 64, 64)
        with torch.no_grad():
            cost = self.model(patches)
        costm = cost.view(11, 23)
        cost = F.interpolate(costm.unsqueeze(0).unsqueeze(0), size=(704, 1472), mode='nearest')
        return cost
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/robodata/haresh92/spot-vrl/models/acc_0.99604_20-01-2023-08-13-17_/cost_model.pt')
    parser.add_argument('--bag_path', type=str, default='/robodata/eyang/data/2023-01-21/2023-01-21-15-41-04.bag')
    parser.add_argument('--output_path', type=str, default='cost_video.mp4', help='path to save the video, including the name w/ extension (.mp4)')
    args = parser.parse_args()
    
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
        cost = (cost * 255.0/6.0).astype(np.uint8)
        cost = cv2.cvtColor(cost, cv2.COLOR_GRAY2RGB)
        cost = cv2.resize(cost, (1472, 704))
        
        stacked_img = cv2.cvtColor(np.hstack((bevimage, cost)), cv2.COLOR_RGB2BGR)
        out.write(stacked_img)
            
        # if ctrl+c is pressed, break the loop, save the video and exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            break

    # out.release()
        
        
        