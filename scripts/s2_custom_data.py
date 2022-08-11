"""Contains the dataloader used by the second model"""
__author__= "Daniel Farkash"
__email__= "dmf248@cornell.edu"
__date__= "August 10, 2022"

import pickle
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from termcolor import cprint
from matplotlib import image
import os
import argparse
from pytorch_lightning import *
import yaml
import time
from PIL import Image
import glob
from tqdm import tqdm
from scipy.signal import periodogram
import pytorch_lightning as pl
from _25_train_jackal import *
from sklearn.preprocessing import StandardScaler
import cv2

class s2CustomDataset(Dataset):

    #file_path must be path leading to directory containing data pickle files (no / at end)
    def __init__(self, file_path, full = True):

        self.file_path = file_path 
        self.full = full

        # Load the inertial data if you want to train using the full data
        if self.full:
            imu_path = file_path +"/inertial_data.pkl"
            cprint('Loading data from {}'.format(imu_path))
            self.imu_data = pickle.load(open(imu_path, 'rb'))

        # The data itself includes some invalid indexes from which not all 25 patches can be extracted, so 
        # we load a dictionary that contains all of the valid indexes so that we know which ones to use
        dict_path = file_path +"/valid_idxs.pkl"

        if os.path.exists(dict_path):

            # load valid index dictionary
            cprint('Loading valid idxs from {}'.format(dict_path))
            self.dict = pickle.load(open(dict_path, 'rb'))

        else:

            # if there is no valid index dictionary, then we go through all of the data and create one ourselves
            self.dict = {}
            count_invalid = 0
            print("Finding all valid indexes in filepath")

            # Ensure that each of the 25 patches has all 10 different perspectives accounted for
            for folder_name in tqdm(glob.glob(file_path + "/*")):
                valid = True
                for sub_folder_name in glob.glob(folder_name + "/*"):
                    count = 0
                    for file_name in glob.glob(sub_folder_name + "/*"):
                        count = count+1
                    if count <10:
                        valid = False

                self.dict[folder_name]=valid

                # keep track of how many invalid indexes are found (less than 5%)
                if not(valid):
                    count_invalid +=1

            print("Found valid indexes in filepath")
            print("Number of invalid indexes: " + str(count_invalid))
            pickle.dump(self.dict, open(dict_path, 'wb'))

    # find the number of (valid) data entries
    def __len__(self):
        if self.full:
            return len(self.imu_data)
        else:
            return len(self.dict) -3

    # retrieve an individual piece of data
    def __getitem__(self, idx):

        idx_path = self.file_path + "/" + str(idx)

        # ensure that index is valid
        if not(self.dict[idx_path]):

            # if not, randomly sample a new index to use instead
            rand_idx= random.randint(0,(self.__len__()))
            
            return self.__getitem__(rand_idx)

        main_patch_path = idx_path + "/10"

        # read randomly sampled perspective for visual patch  
        main_rand_1 = random.randint(0,9)
        main_patch_path_1 = main_patch_path +"/"+str(main_rand_1)+ ".png"
        main_patch_1 = cv2.imread(main_patch_path_1)
        
        # normalize, format data
        main_patch_1 = main_patch_1.astype(np.float32) / 255.0
        main_patch_1 = np.moveaxis(main_patch_1, -1, 0)

        # get fourier transform of inertial data if desired
        if self.full:

            inertial_data = self.imu_data[idx]
            inertial_data = np.asarray(inertial_data).reshape((200, 6))

            # convert to periodogram
            _, acc_x = periodogram(inertial_data[:,0], fs=70)
            _, acc_y = periodogram(inertial_data[:,1], fs=70)
            _, acc_z = periodogram(inertial_data[:,2], fs=70)
            _, gyro_x = periodogram(inertial_data[:,3], fs=70)
            _, gyro_y = periodogram(inertial_data[:,4], fs=70)
            _, gyro_z = periodogram(inertial_data[:,5], fs=70)
            inertial_data = np.hstack((acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)).flatten()
        else:
            inertial_data = np.zeros((1,128))
  
        return main_patch_1, inertial_data

class s2DataLoader(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=32, full = True):

        super(s2DataLoader, self).__init__()

        self.batch_size = batch_size
        self.data_config_path = data_config_path
        self.full = full

        self.setup()
    
    def setup(self, stage=None):

        # combine datasets specified in data config path to make complete trin/val loaders
        with open(self.data_config_path) as file:
            data_config = yaml.full_load(file)

        train_data_path = data_config['train']
        val_data_path = data_config['val']

        self.train_dataset = ConcatDataset([s2CustomDataset(file, self.full) for file in train_data_path])
        self.val_dataset = ConcatDataset([s2CustomDataset(file, self.full) for file in val_data_path])

        print('Train dataset size:', len(self.train_dataset))
        print('Val dataset size:', len(self.val_dataset))


    # Try increasing the number of workers if you think that training is too slow
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True 
               if len(self.train_dataset) % self.batch_size != 0 else False)

    # Must shuffle for imlpementation of softmax to be appropriate
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True 
               if len(self.val_dataset) % self.batch_size != 0 else False)



    # Prevous versions of the dataloader which took in whole models as input (instead of just encoders)
    # or did encoding preprocessing have been deleted 