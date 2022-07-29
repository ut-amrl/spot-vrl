import pickle
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from termcolor import cprint
from matplotlib import image
import os
import argparse
import pytorch_lightning as pl
import yaml
import time
from PIL import Image
import glob
from tqdm import tqdm
from scipy.signal import periodogram

class CustomDataset(Dataset):
    #file_path must be path leading to directory containing imu pickle and other img folders (no / at end)
    def __init__(self, file_path):
        self.file_path = file_path 
        self.label = file_path.split('/')[-2]
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

        idx_path = self.file_path + "/" + str(idx)

        if not(self.dict[idx_path]):
            # print("Invalid patch")
            rand_idx= random.randint(0,self.__len__()-1)
            # print("New idx: "+str(rand_idx))
            
            return self.__getitem__(rand_idx)

        main_patch_path = idx_path + "/10"
        if True:   
            # t0 = time.time()
            # main_rand_1 = random.randint(0,9)
            
            # Choosing patch taken from close to its actual location and not randomizing
            main_rand_1 = 8
            main_patch_path_1 = main_patch_path +"/"+str(main_rand_1)+ ".png"
            main_patch_1 = cv2.imread(main_patch_path_1)
            # main_patch_1 = cv2.resize(main_patch_1, (128, 128))
            main_patch_1 = main_patch_1.astype(np.float32) / 255.0 # normalize
            main_patch_1 = np.moveaxis(main_patch_1, -1, 0)
            # t1 = time.time()
            # total = t1-t0
            # print("image time: " +str(total))

            main_rand_2 = random.choice([i for i in range(0,9) if i not in [main_rand_1]])
            main_patch_path_2 = main_patch_path +"/"+str(main_rand_2)+ ".png"
            main_patch_2 = cv2.imread(main_patch_path_2)
            # main_patch_2 = cv2.resize(main_patch_2, (128, 128))
            main_patch_2 = main_patch_2.astype(np.float32) / 255.0 # normalize
            main_patch_2 = np.moveaxis(main_patch_2, -1, 0)
            
            main_patch_lst = [main_patch_1,main_patch_2]

        patch_list_1 = []
        patch_list_2 = []
        for i in range(25):
            patch_path = idx_path + "/" + str(i)

            if True:
                
                rand_1 = random.randint(0,9)
                patch_path_1 = patch_path +"/"+str(rand_1)+ ".png"
                patch_1 = cv2.imread(patch_path_1)
                # patch_1 = cv2.resize(patch_1, (128, 128))
                patch_1 = patch_1.astype(np.float32) / 255.0 # normalize
                patch_1 = np.moveaxis(patch_1, -1, 0)

                
                rand_2 = random.choice([i for i in range(0,9) if i not in [rand_1]])
                patch_path_2 = patch_path +"/"+str(rand_2)+ ".png"
                patch_2 = cv2.imread(patch_path_2)
                # patch_2 = cv2.resize(patch_2, (128, 128))
                patch_2 = patch_2.astype(np.float32) / 255.0 # normalize
                patch_2 = np.moveaxis(patch_2, -1, 0)

                patch_list_1.append(patch_1)
                patch_list_2.append(patch_2)


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

        # inertial_data = np.expand_dims(inertial_data, axis=0)
  
        return main_patch_lst, inertial_data, patch_list_1, patch_list_2, self.label

class MyDataLoader(pl.LightningDataModule):
	def __init__(self, data_config_path, batch_size=32):
		super(MyDataLoader, self).__init__()
		self.batch_size = batch_size
		self.data_config_path = data_config_path
		self.setup()

	def setup(self, stage=None):
		with open(self.data_config_path) as file:
			data_config = yaml.full_load(file)
   
		train_data_path = data_config['train']
		val_data_path = data_config['val']

		self.train_dataset = ConcatDataset([CustomDataset(file) for file in train_data_path])
		self.val_dataset = ConcatDataset([CustomDataset(file) for file in val_data_path])

		# find mean, std statistics of inertial data in the training set
		# print('Finding mean and std statistics of inertial data in the training set...')
		# tmp = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
		# for _,i,_,_,_ in tmp:
		# 	i = i.numpy()
		# 	break
		# self.inertial_shape = i.shape[1]
		# print('Inertial shape:', self.inertial_shape)
		# print('Data statistics have been found.')
		# del tmp

		print('Train dataset size:', len(self.train_dataset))
		print('Val dataset size:', len(self.val_dataset))

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True if len(self.train_dataset) % self.batch_size != 0 else False)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10, drop_last=True if len(self.val_dataset) % self.batch_size != 0 else False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test custom dataset')
    parser.add_argument('--input', type=str, default='/robodata/dfarkash/patch_images/concrete/_2022-05-10-15-51-08', metavar='N', help='input location')
    parser.add_argument('--out', type=str,default='./',metavar='N',help='save location')
    args = parser.parse_args()
    dataset = CustomDataset(args.input)
    print(dataset.__len__())
    main_patch_lst, inertial_data, patch_list_1, patch_list_2, label = dataset.__getitem__(32)
    print(label)
    print(len(patch_list_1))

    print(main_patch_lst[0])

    # im = Image.fromarray(main_patch_lst[0])
    # img_path = "/robodata/dfarkash/test_data/blk_tst.png"
    # im.save(img_path)

    dm = MyDataLoader('/robodata/dfarkash/spot-vrl/jackal_data/test_config.yaml')