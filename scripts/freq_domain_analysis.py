import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import random

with open('data_tmp/train/cement/2022-03-14-17-36-52.pkl', 'rb') as f:
	cement_data = pickle.load(f)

with open('data_tmp/train_cache/speedway/2022-03-29-18-56-30.pkl', 'rb') as f:
	speedway_data = pickle.load(f)

with open('data_tmp/train_cache/grass/2022-03-14-18-45-10.pkl', 'rb') as f:
	grass_data = pickle.load(f)

with open('data_tmp/train/smallrocks/2022-03-15-10-15-27.pkl', 'rb') as f:
	smallrocks_data = pickle.load(f)


rangeval = list(range(300, min(len(cement_data), len(speedway_data), len(grass_data))-75))
# randomize list
random.shuffle(rangeval)

for i in rangeval:
	cement_freq = np.abs(fftpack.fft(cement_data[i]['linear_velocity'][-26:, 0]))
	cement_freq = cement_freq[len(cement_freq)//2:]
	speedway_freq = np.abs(fftpack.fft(speedway_data[i]['linear_velocity'][-26:, 0]))
	speedway_freq = speedway_freq[len(speedway_freq)//2:]
	grass_freq = np.abs(fftpack.fft(grass_data[i]['linear_velocity'][-26:, 0]))
	grass_freq = grass_freq[len(grass_freq)//2:]
	smallrocks_freq = np.abs(fftpack.fft(smallrocks_data[i]['linear_velocity'][-26:, 0]))
	smallrocks_freq = smallrocks_freq[len(smallrocks_freq)//2:]

	print('cement : ', np.sum(cement_freq))
	print('speedway : ', np.sum(speedway_freq))
	print('grass : ', np.sum(grass_freq))
	print('smallrocks : ', np.sum(smallrocks_freq))
	print('\n')

	plt.plot(cement_freq, label='cement')
	plt.plot(speedway_freq, label='speedway')
	plt.plot(grass_freq, label='grass')
	plt.plot(smallrocks_freq, label='smallrocks')
	plt.legend()
	plt.show()

	# plt.figure()
	# plt.plot(cement_data[i]['linear_velocity'][-26:, 2], label='cement')
	# plt.plot(speedway_data[i]['linear_velocity'][-26:, 2], label='speedway')
	# plt.plot(grass_data[i]['linear_velocity'][-26:, 2], label='grass')
	# plt.plot(smallrocks_data[i]['linear_velocity'][-26:, 2], label='smallrocks')
	# plt.legend()
	# plt.show()

	input()
