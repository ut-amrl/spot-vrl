### TODO list
## Deadline: 01-20-2023


- [X] generate plots to visualize the inertial data collected from the IMU on the spot (analog, fft, PSD)
    - This is done. The scripts can be found in scripts/viz_spot_inertial.py
    - Additionally, using the PSD instead of raw analog values of IMU, feet and leg achieved better results, 
        so scripts/train_naturl_representations.py has been modified to use PSD instead of raw analog values.
         
- [ ] save the k-means clustering results to a file. Use the GUI to get preferences from human, and save the preferences
- [ ] train the cost function network
- [ ] write a wrapper script to wrap the encoder and the costfunction network into a single .pt file
