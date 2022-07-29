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

        if full:
            imu_path = file_path +"/inertial_data.pkl"
            cprint('Loading data from {}'.format(imu_path))
            self.imu_data = pickle.load(open(imu_path, 'rb'))

        dict_path = file_path +"/valid_idxs.pkl"
        if os.path.exists(dict_path):
            cprint('Loading valid idxs from {}'.format(dict_path))
            self.dict = pickle.load(open(dict_path, 'rb'))
        else:
            self.dict = {}
            count_invalid = 0
            print("Finding all valid indexes in filepath")
            # for folder_name in glob.glob(file_path + "/[0-" + str(len(self.imu_data)-1)+"]/[0-24]"):
            for folder_name in tqdm(glob.glob(file_path + "/*")):
                valid = True
                for sub_folder_name in glob.glob(folder_name + "/*"):
                    count = 0
                    for file_name in glob.glob(sub_folder_name + "/*"):
                        count = count+1
                    if count <10:
                        # print(folder_name)
                        valid = False
                self.dict[folder_name]=valid
                if not(valid):
                    count_invalid +=1
            print("Found valid indexes in filepath")
            print("Number of invalid indexes: " + str(count_invalid))
            pickle.dump(self.dict, open(dict_path, 'wb'))


    def __len__(self):
        if self.full:
            return len(self.imu_data)
        else:
            return len(self.dict)

    def __getitem__(self, idx):


        idx_path = self.file_path + "/" + str(idx)

        if not(self.dict[idx_path]):
            # print("Invalid patch")
            rand_idx= random.randint(0,self.__len__()-1)
            # print("New idx: "+str(rand_idx))
            
            return self.__getitem__(rand_idx)

        main_patch_path = idx_path + "/10"
          
        main_rand_1 = random.randint(0,9)
        main_patch_path_1 = main_patch_path +"/"+str(main_rand_1)+ ".png"
        main_patch_1 = cv2.imread(main_patch_path_1)
        # main_patch_1 = cv2.resize(main_patch_1, (128, 128))
        main_patch_1 = main_patch_1.astype(np.float32) / 255.0 # normalize
        main_patch_1 = np.moveaxis(main_patch_1, -1, 0)

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
        with open(self.data_config_path) as file:
            data_config = yaml.full_load(file)

        train_data_path = data_config['train']
        val_data_path = data_config['val']

        self.train_dataset = ConcatDataset([s2CustomDataset(file, self.full) for file in train_data_path])
        self.val_dataset = ConcatDataset([s2CustomDataset(file, self.full) for file in val_data_path])

        print('Train dataset size:', len(self.train_dataset))
        print('Val dataset size:', len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)



class s2ModelCustomDataset(Dataset):
    #file_path must be path leading to directory containing data pickle files (no / at end)
    def __init__(self, file_path, model, visual_encoder, inertial_encoder, preferences):
        # self.file_path = file_path 
        # data_path = file_path +"/data.pkl"
        # cprint('Loading data from {}'.format(data_path))
        # data = pickle.load(open(data_path, 'rb'))
        # self.vis_patch= data['vis_patch']
        # self.clusters= data['clusters']
        # self.elbow= data['elbow']

        # cost_path = file_path +"/cost.pkl"
        # self.costs = pickle.load(open(cost_path, 'rb'))

        # model_path = "/home/dfarkash/cost_data/model.pkl"
        # # model_path = "/home/dfarkash/reserve/model2.pkl"
        # cprint('Loading cluster_model from {}'.format(model_path))
        # self.model = pickle.load(open(model_path, 'rb'))

        # # print(self.model.labels_)

        # if True:
        #     preferences= [.7,.1,.2,.3,.4,.5,.6]
        #     with open("/home/dfarkash/cost_data/preferences.pkl", "wb") as f:
        #             pickle.dump(preferences, f)

        # preferences_path = "/home/dfarkash/cost_data/preferences.pkl"
        # cprint('Loading user preferences from {}'.format(preferences_path))
        # self.preferences = pickle.load(open(preferences_path, 'rb'))

        # net_path = "/home/dfarkash/spot-vrl/models/26-07-2022-11-06-47_.ckpt"
        # cprint('Loading model from {}'.format(net_path))
        # net = BarlowModel.load_from_checkpoint(net_path)
        # net.eval()
        # # self.visual_encoder = net.visual_encoder
        # # self.inertial_encoder = net.inertial_encoder
        # self.net = net

        # visual_encoder = BarlowModel().visual_encoder
        # visual_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/visual_encoder.pt"))
        # visual_encoder.eval()
        # self.visual_encoder = visual_encoder

        # inertial_encoder = BarlowModel().inertial_encoder
        # inertial_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/inertial_encoder.pt"))
        # inertial_encoder.eval()
        # self.inertial_encoder = inertial_encoder

        self.model = model
        self.visual_encoder = visual_encoder
        self.inertial_encoder = inertial_encoder
        self.preferences = preferences

        self.file_path = file_path 

        imu_path = file_path +"/inertial_data.pkl"
        cprint('Loading data from {}'.format(imu_path))
        self.imu_data = pickle.load(open(imu_path, 'rb'))

        dict_path = file_path +"/valid_idxs.pkl"
        if os.path.exists(dict_path):
            cprint('Loading valid idxs from {}'.format(dict_path))
            self.dict = pickle.load(open(dict_path, 'rb'))
        else:
            self.dict = {}
            count_invalid = 0
            print("Finding all valid indexes in filepath")
            # for folder_name in glob.glob(file_path + "/[0-" + str(len(self.imu_data)-1)+"]/[0-24]"):
            for folder_name in tqdm(glob.glob(file_path + "/*")):
                valid = True
                for sub_folder_name in glob.glob(folder_name + "/*"):
                    count = 0
                    for file_name in glob.glob(sub_folder_name + "/*"):
                        count = count+1
                    if count <10:
                        # print(folder_name)
                        valid = False
                self.dict[folder_name]=valid
                if not(valid):
                    count_invalid +=1
            print("Found valid indexes in filepath")
            print("Number of invalid indexes: " + str(count_invalid))
            pickle.dump(self.dict, open(dict_path, 'wb'))


    def __len__(self):
        return len(self.imu_data)

    def __getitem__(self, idx):

        # cv2.setNumThreads(0)

        # cluster = self.clusters[idx]
        # vis_patch = self.vis_patch[idx]
        # cost = self.costs[cluster]


        idx_path = self.file_path + "/" + str(idx)

        if not(self.dict[idx_path]):
            # print("Invalid patch")
            rand_idx= random.randint(0,self.__len__()-1)
            # print("New idx: "+str(rand_idx))
            
            return self.__getitem__(rand_idx)

        main_patch_path = idx_path + "/10"
          
        main_rand_1 = random.randint(0,9)
        main_patch_path_1 = main_patch_path +"/"+str(main_rand_1)+ ".png"
        main_patch_1 = cv2.imread(main_patch_path_1)
        # main_patch_1 = cv2.resize(main_patch_1, (128, 128))
        main_patch_1 = main_patch_1.astype(np.float32) / 255.0 # normalize
        main_patch_1 = np.moveaxis(main_patch_1, -1, 0)

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

        # visual_encoding = self.net.visual_encoder(input = torch.from_numpy(main_patch_1).float(), weight = self.net.visual_encoder.weight, bias = self.net.visual_encoder.bias)
        # inertial_encoding = self.net.inertial_encoder(torch.from_numpy(inertial_data).float())

        # net = self.net

        # vis = torch.from_numpy(np.expand_dims(np.array([main_patch_1]),1)).float()
        # inert=torch.from_numpy(np.moveaxis(np.expand_dims(inertial_data,1),-1,0)).float()

        with torch.no_grad():
            # visual_encoding, inertial_encoding, ls1, ls2 = net(vis,inert,s1=False)
            visual_encoding = self.visual_encoder(input = torch.from_numpy(np.expand_dims(main_patch_1,0)).float())
            # print(torch.from_numpy(np.expand_dims(main_patch_1,0)).float().shape)
            inertial_encoding = self.inertial_encoder(torch.from_numpy(np.moveaxis(np.expand_dims(inertial_data,1),-1,0)).float())
        # print(visual_encoding.shape)
        # print(inertial_encoding.shape)

        data = torch.cat((visual_encoding, inertial_encoding), dim=1)

        scaler = StandardScaler()
        data=data.cpu()
        data=data.numpy()
        # data = scaler.fit_transform(data)
        
        # print(data.shape)
        cluster = self.model.predict(data)

        # print('inside get item')

        # print(cluster)

        cost = self.preferences[cluster[0]]

        
  
        return main_patch_1, np.float32(cost)

class s2ModelDataLoader(pl.LightningDataModule):
    def __init__(self, data_config_path, batch_size=32):
        super(s2ModelDataLoader, self).__init__()
        self.batch_size = batch_size
        self.data_config_path = data_config_path
        
        model_path = "/home/dfarkash/cost_data/model.pkl"
        # model_path = "/home/dfarkash/reserve/model2.pkl"
        cprint('Loading cluster_model from {}'.format(model_path))
        self.model = pickle.load(open(model_path, 'rb'))

        # print('just checking the model')
        # tmp = np.zeros((1, 256)).astype(np.float32)
        # print(self.model.predict(tmp))
        # print('it works')

        if False:
            preferences= [.7,.1,.2,.3,.4,.5,.6]
            with open("/home/dfarkash/cost_data/preferences.pkl", "wb") as f:
                    pickle.dump(preferences, f)

        preferences_path = "/home/dfarkash/cost_data/preferences.pkl"
        cprint('Loading user preferences from {}'.format(preferences_path))
        self.preferences = pickle.load(open(preferences_path, 'rb'))

        visual_encoder = BarlowModel().visual_encoder
        visual_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/visual_encoder.pt"))
        visual_encoder.eval()
        self.visual_encoder = visual_encoder

        inertial_encoder = BarlowModel().inertial_encoder
        inertial_encoder.load_state_dict(torch.load("/home/dfarkash/cost_data/inertial_encoder.pt"))
        inertial_encoder.eval()
        self.inertial_encoder = inertial_encoder


        self.setup()
    
    def setup(self, stage=None):
        with open(self.data_config_path) as file:
            data_config = yaml.full_load(file)

        train_data_path = data_config['train']
        val_data_path = data_config['val']

        self.train_dataset = ConcatDataset([s2ModelCustomDataset(file, self.model, self.visual_encoder, self.inertial_encoder, self.preferences) for file in train_data_path])
        self.val_dataset = ConcatDataset([s2ModelCustomDataset(file, self.model, self.visual_encoder, self.inertial_encoder, self.preferences) for file in val_data_path])

        print('Train dataset size:', len(self.train_dataset))
        print('Val dataset size:', len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test custom dataset')
    parser.add_argument('--input', type=str, default='/home/dfarkash/patch_images/concrete/_2022-05-10-16-07-18', metavar='N', help='input location')
    args = parser.parse_args()
    dataset = CustomDataset(args.input)
    print(dataset.__len__())
    patch, cost = dataset.__getitem__(138)
    print(patch)
    print(cost)

    # im = Image.fromarray(main_patch_lst[0])
    # img_path = "/robodata/dfarkash/test_data/blk_tst.png"
    # im.save(img_path)

    dm = MyDataLoader('/home/dfarkash/spot-vrl/jackal_data/different.yaml')