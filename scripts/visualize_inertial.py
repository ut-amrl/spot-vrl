import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':
	with open('data/train/cement/2022-03-14-17-36-52.pkl', 'rb') as f:
		data = pickle.load(f)
	print('Successfully loaded data.')

	tmp = []
	for i in range(len(data)):
		tmp.append(data[i]['inertial'][-13:, :].flatten())
	tmp = np.asarray(tmp)

	mean = np.mean(tmp, axis=0)
	std = np.std(tmp, axis=0)

	print('Mean:', mean)
	print('Std:', std)
