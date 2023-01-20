#!/usr/env bash

# print the number of python processes for the current user
ps -ef | grep python | grep -v grep | grep $USER | wc -l

# kill all python processes for the current user
ps -ef | grep python | grep -v grep | grep $USER | awk '{print $2}' | xargs kill -9

# done
echo "Done"