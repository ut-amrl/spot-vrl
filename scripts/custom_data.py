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

class CustomDataset(Dataset):
    #file_path must be path leading to directory containing imu pickle and other img folders (no / at end)
    def __init__(self, file_path):
        self.file_path = file_path 
        self.label = file_path.split('/')[-2]
        imu_path = file_path +"/inertial_data.pkl"
        cprint('Loading data from {}'.format(imu_path))
        self.imu_data = pickle.load(open(imu_path, 'rb'))

    def __len__(self):
        return len(self.imu_data)

    def __getitem__(self, idx):

        idx_path = self.file_path + "/" + str(idx)

        main_patch_path = idx_path + "/10"

        main_lst = os.listdir(main_patch_path)
        num_entries = len(main_lst)


        if num_entries < 2:
            print("No main patch")
            rand_idx= random.randint(0,self.__len__()-1)
            print("New idx: "+str(rand_idx))
            
            return self.__getitem__(rand_idx)
        else:
            
            main_rand_1 = random.randint(0,num_entries-1)
            main_patch_path_1 = main_patch_path +"/"+str(main_rand_1)+ ".png"
            main_patch_1 = image.imread(main_patch_path_1)
            main_patch_1 = cv2.resize(main_patch_1, (128, 128))
            main_patch_1 = main_patch_1.astype(np.float32) / 255.0 # normalize
            main_patch_1 = np.moveaxis(main_patch_1, -1, 0)

            main_rand_2 = random.choice([i for i in range(0,num_entries-1) if i not in [main_rand_1]])
            main_patch_path_2 = main_patch_path +"/"+str(main_rand_2)+ ".png"
            main_patch_2 = image.imread(main_patch_path_2)
            main_patch_2 = cv2.resize(main_patch_2, (128, 128))
            main_patch_2 = main_patch_2.astype(np.float32) / 255.0 # normalize
            main_patch_2 = np.moveaxis(main_patch_2, -1, 0)
            
            main_patch_lst = [main_patch_1,main_patch_2]

        patch_list_1 = []
        patch_list_2 = []
        for i in range(25):
            patch_path = idx_path + "/" + str(i)

            lst = os.listdir(patch_path)
            num_entries = len(lst)

            if num_entries >= 2:
                
                rand_1 = random.randint(0,num_entries-1)
                patch_path_1 = patch_path +"/"+str(rand_1)+ ".png"
                patch_1 = image.imread(patch_path_1)
                patch_1 = cv2.resize(patch_1, (128, 128))
                patch_1 = patch_1.astype(np.float32) / 255.0 # normalize
                patch_1 = np.moveaxis(patch_1, -1, 0)

                
                rand_2 = random.choice([i for i in range(0,num_entries-1) if i not in [rand_1]])
                patch_path_2 = patch_path +"/"+str(rand_2)+ ".png"
                patch_2 = image.imread(patch_path_2)
                patch_2 = cv2.resize(patch_2, (128, 128))
                patch_2 = patch_2.astype(np.float32) / 255.0 # normalize
                patch_2 = np.moveaxis(patch_2, -1, 0)

                patch_list_1.append(patch_1)
                patch_list_2.append(patch_2)
            else:
                valid = False
                while valid == False:
                    print("No patch "+str(i))
                    fix_rand_idx= random.randint(0,self.__len__()-1)
                    print("New idx: "+str(fix_rand_idx))
                    fix_idx_path = self.file_path + "/" + str(fix_rand_idx)
                    fix_patch_path = fix_idx_path + "/" + str(i)

                    fix_lst = os.listdir(fix_patch_path)
                    fix_num_entries = len(fix_lst)
                    if fix_num_entries >=2:
                        valid = True

                rand_1 = random.randint(0,fix_num_entries-1)
                patch_path_1 = fix_patch_path +"/"+str(rand_1)+ ".png"
                patch_1 = image.imread(patch_path_1)
                patch_1 = cv2.resize(patch_1, (128, 128))
                patch_1 = patch_1.astype(np.float32) / 255.0 # normalize
                patch_1 = np.moveaxis(patch_1, -1, 0)

                
                rand_2 = random.choice([i for i in range(0,fix_num_entries-1) if i not in [rand_1]])
                patch_path_2 = fix_patch_path +"/"+str(rand_2)+ ".png"
                patch_2 = image.imread(patch_path_2)
                patch_2 = cv2.resize(patch_2, (128, 128))
                patch_2 = patch_2.astype(np.float32) / 255.0 # normalize
                patch_2 = np.moveaxis(patch_2, -1, 0)

                patch_list_1.append(patch_1)
                patch_list_2.append(patch_2)




        inertial_data = self.imu_data[idx]
        inertial_data = np.expand_dims(inertial_data, axis=0)
  
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
		print('Finding mean and std statistics of inertial data in the training set...')
		tmp = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
		for _,i,_,_,_ in tmp:
			i = i.numpy()
			break
		self.inertial_shape = i.shape[1]
		print('Inertial shape:', self.inertial_shape)
		print('Data statistics have been found.')
		del tmp

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

    dm = MyDataLoader('/robodata/dfarkash/spot-vrl/jackal_data/test_config.yaml')