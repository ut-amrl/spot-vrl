"""
There is an error in the training. the pebbled sidewalk is being provided a higher than usual cost.
Investigate if the dataset is noisy by generating a video of all the rosbags.
"""

import rosbag as rosbagapi
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
    
if __name__ == '__main__':
    
    # read all the bag names in spot_data/bag_names.txt
    bag_names = []
    with open('./spot_data/bag_names.txt', 'r') as f:
        for line in f:
            bag_names.append(line.strip())
    for bag_name in bag_names:
        bag_path = '/robodata/eyang/data/' + bag_name.split('.')[0][:10] + '/' + bag_name
        output_path = './spot_data/' + bag_name.replace('.bag', '.mp4')
        
        cprint('Processing bag: ' + bag_name, 'green')
        cprint('Output path: ' + output_path, 'green')
    
        rosbag = rosbagapi.Bag(bag_path)
        
        # find number of frames in the rosbag
        numframes = rosbag.get_message_count(topic_filters=['/bev/single/compressed'])
        
        stacked_img_list = []
        
        # save all the stacked images in a list and save in disk as a MPEG-4 video
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 10, (1476, 749))

        # use tqdm and rosbag to iterate over the frames
        for topic, msg, t in tqdm(rosbag.read_messages(topics=['/bev/single/compressed']), total=numframes):
            curr_bev_img = np.fromstring(msg.data, np.uint8)
            curr_bev_img = cv2.imdecode(curr_bev_img, cv2.IMREAD_COLOR)
            
            out.write(curr_bev_img.copy())
                
            # if ctrl+c is pressed, break the loop, save the video and exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                break
        
        out.release()
        