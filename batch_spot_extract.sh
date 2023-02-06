#!/usr/bin/env bash

# This script reads the rosbag names from the file "bag_names.txt" and runs the 
# spot_extract.sh file with the argument 1 as <bag_name> for each bag
# the location for the rosbag is in the name of the rosbag itself

# read the bag names from the file, name of the file is the first argument
# if no argument is given, the default file name is "bag_names.txt"
if [ -z "$1" ]
then
    bag_names_file="spot_data/bag_names.txt"
else
    bag_names_file=$1
fi

# read the bag names from the file
bag_names=$(cat $bag_names_file)

# run the spot_extract.sh script for each bag
# run it parallelly for all rosbags. everytime the python script is called, 
# it runs in an independent tmux session
# if the rosbag is named 2022-12-16-12-02-29.bag, then it 
# is in the folder /robodata/eyang/data/2022-12-16/2022-12-16-12-02-29.bag
# second argument is the total samples to extract from the rosbag

total_samples=3000
for bag_name in $bag_names
do 
    # tmux session name current time
    tmux_session_name=$(date +%s)
    echo "Launching rosbag $bag_name in tmux session $tmux_session_name"
    bag_path="/robodata/eyang/data/${bag_name:0:10}/${bag_name}"
    tmux new-session -d -s $tmux_session_name "bash spot_extract.sh $bag_path $total_samples"
    sleep 1
done
echo "Launched all the rosbags in tmux sessions"
