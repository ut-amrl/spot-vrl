# spot-vrl
This is the most recently updated branch of the project.

When creating the environment on robovision, source Haresh's spot-vrl conda environment.

  Commands:

    source /robodata/haresh92/conda/bin/activate

    conda activate spot-vrl
    
 You cannot visualize anything on Robovision, so a good way to go about visualizing any of the projections/plots is to use tensorboard through vscode or to run this commnd locally:
 
    tensorboard --logdir lightning_logs
  
 If you wish to run patch extraction on robovision, do this:
 
 ONLY ONCE:
 
        catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m

 EVERY TIME you ssh in:

        source /opt/ros/noetic/setup.bash

        source ./devel/setup.bash

        source /robodata/haresh92/conda/bin/activate

        conda activate spot-vrl

        PYTHONPATH="/robodata/haresh92/conda/envs/spot-vrl/lib/python3.8/site-packages:$PYTHONPATH"

RUN COMMAND:

        roslaunch launch/parse_rosbag.launch rosbag_path:=/robodata/dfarkash/test_data/sample.bag save_data_path:=/robodata/dfarkash/test_data/ visualize:=false
        
 -> Makesure to change the paths so that they are appropriate
 -> If you want to see the visualization (visualization:=true) you will have to do this locally

