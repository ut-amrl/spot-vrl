import numpy as np

def process_feet_data(feet):
    filtered_feet = []
    for fi in range(feet.shape[0]):
        contacts = [feet[fi, 3], feet[fi, 6+3], feet[fi, 12+3], feet[fi, 18+3]]
        mu, std = [feet[fi, 4], feet[fi, 10], feet[fi, 16], feet[fi, 22]], [feet[fi, 5], feet[fi, 11], feet[fi, 17], feet[fi, 23]]
        
        for i in range(4):
            if contacts[i] != 1:
                mu[i], std[i] = -1, -1
                    
        # np remove all mu and std values from feet[fi, :]
        curr = np.delete(feet[fi, :], [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23])
        curr = np.hstack((curr, mu, std))
        
        filtered_feet.append(curr)
    return  np.asarray(filtered_feet)