# spot-vrl
This is the most recently updated branch of the project.

The code used during PATCH EXTRACTION is found in 

    launch/parse_rosbag.launch
    
    src/lidar_rosbag_parser/scripts/rosbag_data_rerecorder.py

Usually, the configurations used when running the models are stored in 

    jackal_data
    
Sample image grids are contained in

    image_grids

The vast majority of the code is in scripts:

Contains the first part of the model (creating representation, saving resulting k-means model and encoders):
    
    _25_train_jackal.py

Contains the dataloader used by the first model:

    dict_custom_data.py
    
Contains the second model (using user preferences, saved items from part 1 to learn cost network):

    s2_train_jackal.py

Contains the dataloader used by the second model:

    s2_custom_data.py
    
Contains the k-means model creation and accuracy calculations used:

    cluster_jackal.py
    
Runs the graphical user interface for getting user preferences (must be run locally using pre-saved image grids - made by the first model after run cimpletion):

    tkinter_gui.py
    
Most of the other files are scripts that were used in the past to automate tests (end in script) or previous versions of the components above. Of these, l1_only and l2_only are versions of model 1 that only use the l1 and l2 losses independently instead of together.
    

When creating the ENVIRONMENT on robovision, source Haresh's spot-vrl conda environment. (do not use poetry)

  Commands:

    source /robodata/haresh92/conda/bin/activate

    conda activate spot-vrl
    
 You cannot visualize anything on Robovision, so a good way to go about VISUAIZING any of the projections/plots is to use tensorboard through vscode or to run this commnd locally:
 
    tensorboard --logdir lightning_logs
  
 If you wish to run PATCH EXTRACTION on robovision, do this:
 
 Only Once:
 
        catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m

 Every Time you ssh in:

        source /opt/ros/noetic/setup.bash

        source ./devel/setup.bash

        source /robodata/haresh92/conda/bin/activate

        conda activate spot-vrl

        PYTHONPATH="/robodata/haresh92/conda/envs/spot-vrl/lib/python3.8/site-packages:$PYTHONPATH"

Run Command:

        roslaunch launch/parse_rosbag.launch rosbag_path:=/robodata/dfarkash/test_data/sample.bag save_data_path:=/robodata/dfarkash/test_data/ visualize:=false
        
 -> Makesure to change the paths so that they are appropriate
 
 -> If you want to see the visualization (visualization:=true) you will have to do this locally
 
 
When RUNNING CODE FOR MODELS 1 and 2, make sure to read the argument descriptions to see what they do. The commands will look something like:

Model 1:

    python scripts/_25_train_jackal.py --dataset_config_path jackal_data/different.yaml --num_gpus 2 --epochs 151 --save 1 --imu_in_rep 1
    
-> Be wary of using more than 2 gpus as it may use up too much memory
-> Be wary when saving, beause data may be overwritten (see location saved to in code)

Model 2:

    python scripts/s2_train_jackal.py --dataset_config_path jackal_data/different.yaml --epochs 151 --full 0 --model_folder _98_save --num_gpus 4
    

Checkout /home/dfarkash for special saves:

patch_images contains all of the data used for training

/home/dfarkash/spot-vrl/jackal_data/different.yaml contains the training/ validation split of the data files which have been used for most experiments.

_98_save and _93_save contain full model 1 (_25_train_jackal) saves (k-means, encoders) for use as model_folder for when training model 2 (s2_train_jackal) whike vis_cost_data and no_inert_rep_cost_data contain model 1's trained using only visual data and withour inertial data for making the representation(respectively). cost_data is the default save location for the k-means and inertial encoders



