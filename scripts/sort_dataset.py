"""
a script that reads the yaml file spot_data/data_config.yaml and sorts the dataset into train, and test folders
"""

import yaml
import os

terrains = ['concrete', 'asphalt', 'grass', 'marble_rock', 'yellow_brick', 'red_brick', 'pebble_pavement', 'mulch', 'bush']

# check if the train and test folders and subfolders exis
if not os.path.exists("spot_data/train"):
    os.mkdir("spot_data/train")
    for terrain in terrains:
        os.mkdir("spot_data/train/" + terrain)
    
if not os.path.exists("spot_data/test"):
    os.mkdir("spot_data/test")
    for terrain in terrains:
        os.mkdir("spot_data/test/" + terrain)
        
        
def move_to_path(processed_path):
    final_path = processed_path
    current_path = "./spot_data/" + final_path.split("/")[-1]
    
    if not os.path.exists(current_path): 
        print("file {} does not exist".format(processed_path))
        return
    
    # move the file from current path to final path
    print("moving file from {} to {}".format(current_path, final_path))
    os.rename(current_path, final_path)
    

# read the yaml file
with open("spot_data/data_config.yaml", "r") as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)
    # process train data rosbags
    for processed_path in data_config['train']:
        move_to_path(processed_path)
    for processed_path in data_config['val']:
        move_to_path(processed_path)
    