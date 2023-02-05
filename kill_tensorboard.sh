#!/usr/env bash


# this script kills all the tensorboard processes
# it is useful when you have multiple tensorboard processes running

# get the pids of all the tensorboard processes
pids=$(ps -ef | grep tensorboard | grep -v grep | awk '{print $2}')

# kill all the tensorboard processes
for pid in $pids
do
    echo "killing tensorboard process with pid $pid"
    kill -9 $pid
done

# check if there are any tensorboard processes running
ps -ef | grep tensorboard | grep -v grep

# if there are no tensorboard processes running, the output should be empty
