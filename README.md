# spot-vrl
This is the most recently updated branch of the project.

The code used during PATCH EXTRACTION is found in 

    launch/parse_rosbag.launch
    
    src/lidar_rosbag_parser/scripts/rosbag_data_rerecorder.py

Usually, the configurations used when running the models are stored in 

    jackal_data
    
Sample image grids are contained in

    image_grids
    
    

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
 


