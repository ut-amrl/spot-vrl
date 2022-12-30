# visualize the inertial, feet and leg data from the spot robot
import matplotlib.pyplot as plt
import glob
import yaml, pickle
from termcolor import cprint
import numpy as np
from scipy import fftpack
from scipy.signal import periodogram

from scripts.utils import process_feet_data


FEET_TOPIC_RATE = 24.0
LEG_TOPIC_RATE = 24.0
IMU_TOPIC_RATE = 200.0

data_config_path = 'spot_data/data_config.yaml'
data_config = yaml.load(open(data_config_path, 'r'), Loader=yaml.FullLoader)
data_config_path = '/'.join(data_config_path.split('/')[:-1])

all_data = {}

data_statistics = pickle.load(open(data_config_path + '/data_statistics.pkl', 'rb'))


for pickle_files_root in data_config['train']:
    pickle_files_paths = glob.glob(pickle_files_root + '/*.pkl')
    label = pickle_files_root.split('/')[-2]
    
    if label not in list(all_data.keys()):
        all_data[label] = {}
    else: 
        cprint('label {} already exists'.format(label), 'red')
        continue
    
    imu_data, leg_data, feet_data = [], [], []
    
    
    for idx in range(len(pickle_files_paths)):
        with open(pickle_files_paths[idx], 'rb') as f:
            data = pickle.load(f)
            
        patches, imu, feet, leg = data['patches'], data['imu'], data['feet'], data['leg']
        
        imu = np.asarray(imu[:, :-3])
        feet = process_feet_data(feet)
        
        # normalize the imu data
        imu = (imu - data_statistics['mean']['imu']) / (data_statistics['std']['imu'] + 1e-7)
        
        leg = (leg - data_statistics['mean']['leg']) / (data_statistics['std']['leg'] + 1e-7)
        
        feet = (feet - data_statistics['mean']['feet']) / (data_statistics['std']['feet'] + 1e-7)
        
        # convert to fft. dim 0 is time, dim 1 is imu. 
        # imu =  np.abs(fftpack.fft(imu, axis=0))
        # imu = imu[:imu.shape[0]//2, :]
        
        # convert to psd. dim 0 is time, dim 1 is imu.
        imu = periodogram(imu, fs=IMU_TOPIC_RATE, axis=0)[1]
        
        leg = periodogram(leg, fs=LEG_TOPIC_RATE, axis=0)[1]
        
        feet = periodogram(feet, fs=FEET_TOPIC_RATE, axis=0)[1]
        
        imu_data.append(imu)
        leg_data.append(leg)
        feet_data.append(feet)
        
    imu_data = np.array(imu_data)
    leg_data = np.array(leg_data)
    feet_data = np.array(feet_data)
    
    all_data[label]['imu'] = imu_data
    all_data[label]['leg'] = leg_data
    all_data[label]['feet'] = feet_data
    print('label: {}'.format(label))
    print('imu data shape: {}'.format(imu_data.shape))
    print('leg data shape: {}'.format(leg_data.shape))
    print('feet data shape: {}'.format(feet_data.shape))

print(all_data.keys())
    
# find mean and std of the data
for label in all_data.keys():
    all_data[label]['imu'] = {
        'mean' : np.mean(all_data[label]['imu'], axis=0),
        'std' : np.std(all_data[label]['imu'], axis=0)
    }
    all_data[label]['leg'] = {
        'mean' : np.mean(all_data[label]['leg'], axis=0),
        'std' : np.std(all_data[label]['leg'], axis=0)
    }
    all_data[label]['feet'] = {
        'mean' : np.mean(all_data[label]['feet'], axis=0),
        'std' : np.std(all_data[label]['feet'], axis=0)
    }
    

# fig, axs = plt.subplots(7, 1, figsize=(10, 10))
# # plot the data
# for label in all_data.keys():
#     for i in range(7):
#         # for ns in range(all_data[label]['imu'].shape[0]):
#         #     axs[i].plot(all_data[label]['imu'][ns, :, i])
#         axs[i].plot(all_data[label]['imu']['mean'][:, i], label=label)
#         axs[i].set_title('imu {}'.format(i))
#         axs[i].set_xlabel('time')
#         axs[i].set_ylabel('value')
#         if i==6:
#             axs[i].legend()
        
#     # save the figure
#     fig.savefig('images/imu.png'.format(label))



# fig, axs = plt.subplots(7, 1, figsize=(10, 10))
# # plot the data
# for label in all_data.keys():
#     for i in range(7):
#         # for ns in range(all_data[label]['imu'].shape[0]):
#         #     axs[i].plot(all_data[label]['imu'][ns, :, i])
#         axs[i].plot(all_data[label]['leg']['mean'][:, i], label=label)
#         axs[i].set_title('leg {}'.format(i))
#         axs[i].set_xlabel('time')
#         axs[i].set_ylabel('value')
#         if i==6:
#             axs[i].legend()
        
#     # save the figure
#     fig.savefig('images/leg.png'.format(label))


fig, axs = plt.subplots(7, 1, figsize=(10, 10))
# plot the data
for label in all_data.keys():
    for i in range(7):
        # for ns in range(all_data[label]['imu'].shape[0]):
        #     axs[i].plot(all_data[label]['imu'][ns, :, i])
        axs[i].plot(all_data[label]['feet']['mean'][:, i], label=label)
        axs[i].set_title('feet {}'.format(i))
        axs[i].set_xlabel('time')
        axs[i].set_ylabel('value')
        if i==6:
            axs[i].legend()
        
    # save the figure
    fig.savefig('images/feet.png'.format(label))


    
            
            

