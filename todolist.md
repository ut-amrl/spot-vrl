### TODO list
## Deadline: 01-20-2023


- [X] generate plots to visualize the inertial data collected from the IMU on the spot (analog, fft, PSD)
    - The scripts can be found in scripts/viz_spot_inertial.py
    - We are now using the PSD instead of raw analog values of IMU, feet and leg achieved better results, so scripts/train_naturl_representations.py has been modified to use PSD instead of raw analog values.

- [X] save the k-means clustering results to a file. Use the GUI to get preferences from human, and save the preferences
    - Skipped the GUI part, but the preferences are now hardcoded in the scripts/train_naturl_cost.py
    - To manually get the preferences, look at the saved 25 patch grids and assign the preferences in scripts/train_naturl_cost.py

- [X] train the cost function network
    - We now have trained cost function network for the following 8 terrains : [asphalt, bush, cement, dark_tile, grass, marble_rock, pebble_pavement, red_brick]
    - The trained cost function network with 99% accuracy can be found in the folder : models/acc_0.99979_best

- [X] write a wrapper script to wrap the encoder and the costfunction network into a single .pt file
    - We are saving the wrapped model in scripts/train_naturl_cost.py
    - scripts/plot_naturl_cost.py contains scripts that can be used to plot the cost function network, using the wrapped model

- [X] QOL Improvement- Make the NATURL representation learning script (cost/train_naturl_representations.py) train parallely on multiple GPUs.
    - Now can train both representations and cost function network parallely on multiple GPUs.
    
- [X] Design the experiments. Come up with a list of training terrains and the scenarios we will need to train and test the NATURL algorithm on the spot. 
- [X] Train the encoder and cost function network for the appropriate terrains and sync with Elvin for the actual experiments on the spot.
- [ ] Start writing the pre-writing form for this paper. Start writing the Abstract and Introduction, Related Work and Experimental Setup sections.

- [ ] Start training on hazard / pepi machines


