import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Dataset, ConcatDataset
import yaml
from tqdm import tqdm
import pickle

from scripts.train_naturl_representations import NATURLDataModule
from scripts.train_rca_cost import RCATerrainDataset
from scripts.models import VisualEncoderModel, CostNet

if __name__ == '__main__':
    
    # data_config_path = '/robodata/haresh92/spot-vrl/spot_data/data_config.yaml'
    # dm = NATURLDataModule(data_config_path, psd=False, data_statistics=None)
    # dataset = dm.val_dataset
    data_config  = yaml.load(open('/robodata/haresh92/spot-vrl/spot_data/data_config.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = ConcatDataset([RCATerrainDataset(pickle_files_root, img_augment=False) for pickle_files_root in data_config['train']])
    
    # load the data_statistics.pkl file
    data_statistics = pickle.load(open('/robodata/haresh92/spot-vrl/spot_data/data_statistics.pkl', 'rb'))
    imin, imax = data_statistics['min']['imu'], data_statistics['max']['imu']
    
    data = {}
    
    for patch1, patch2, imu, leg, feet, label, idx in tqdm(dataset):
        # print(' imu shape :', imu.shape)
        # convert to torch tensor and add batch dimension
        patch1 = torch.tensor(patch1).unsqueeze(0)
        patch2 = torch.tensor(patch2).unsqueeze(0)
        imu = torch.tensor(imu).unsqueeze(0)
        leg = torch.tensor(leg).unsqueeze(0)
        feet = torch.tensor(feet).unsqueeze(0)
        
        # move to device
        patch1 = patch1.cuda()
        patch2 = patch2.cuda()
        imu = imu.cuda()
        leg = leg.cuda()
        feet = feet.cuda()
        
        # forward pass
        # cost = model(patch1.float())
        cost = torch.mean(imu[:, 0]) + torch.mean(imu[:, 1]) + torch.mean(imu[:, 5])
        cost = cost.detach().cpu().numpy().flatten()[0]
        
        if label not in data:
            data[label] = []
        data[label].append(cost)
        
        # if idx > 500:
        #     break
    
    
    # plot the labels in x axis and the mean cost with std in y axis
    import matplotlib.pyplot as plt
    labels = []
    mean_costs = []
    std_costs = []
    for label in data:
        labels.append(label)
        mean_costs.append(np.mean(data[label]))
        std_costs.append(np.std(data[label]))
    
    # find mean across all labels
    for label in data:
        data[label] = np.mean(data[label])
        
    # minmax normalize the costs
    minlabel, maxlabel = min(data.values()), max(data.values())
    for label in data:
        data[label] = (data[label] - minlabel) / (maxlabel - minlabel)
        
    # save the data to a pickle file
    pickle.dump(data, open('rca_costs.pkl', 'wb'))
        
    # plot
    plt.figure()
    plt.bar(data.keys(), data.values())
    plt.xticks(list(data.keys()), rotation=45)
    plt.ylabel('Cost')
    plt.title('Costs for different labels')
    # prevent the labels from being cut off
    plt.tight_layout()
    plt.savefig('rca_costs_minmax.png')
    
    
        
        
        
        
    