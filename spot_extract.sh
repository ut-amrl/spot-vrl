#!/usr/bin/env bash

echo "Sourcing the conda environment"
source /robodata/haresh92/conda/bin/activate spot-vrl

echo "Launching the script to extract the data"
python scripts/spot_extract_data.py -b $1 -n $2