import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':
	with open('data/train/cement/2022-03-14-17-36-52.pkl', 'rb') as f:
		data = pickle.load(f)
	print('Successfully loaded data.')

	# tmp = []
	# for i in range(len(data)):
	# 	tmp.append(data[i]['linear_velocity'][-13:, :].flatten())
	# tmp = np.asarray(tmp)
	#
	# mean = np.mean(tmp, axis=0)
	# std = np.std(tmp, axis=0)
	#
	# print('Mean:', mean)
	# print('Std:', std)

	with open('data/train/cement/2022-03-14-17-36-52.pkl', 'rb') as f:
		data = pickle.load(f)

	print(data[0].keys())



	# create 4x1 subplots
	fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
	ax[0].plot(data[1200]['linear_velocity'][-13:, 0], 'b')
	ax[1].plot(data[1200]['linear_velocity'][-13:, 1], 'b')
	ax[2].plot(data[1200]['linear_velocity'][-13:, 2], 'b')
	# ax[3].plot(data[1200]['linear_velocity'][-13:, 46], 'b')

	# plt.figure()
	# fft_0 = np.fft.fft(data[1200]['linear_velocity'][-13:, 2].flatten())
	# plt.plot(np.abs(fft_0))
	# plt.show()

	with open('data/train/grass/2022-03-14-18-45-10.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['linear_velocity'][-13:, 0], 'g')
	ax[1].plot(data[1200]['linear_velocity'][-13:, 1], 'g')
	ax[2].plot(data[1200]['linear_velocity'][-13:, 2], 'g')
	# ax[3].plot(data[1200]['linear_velocity'][-13:, 46], 'g')

	with open('data/train/smallrocks/2022-03-15-10-15-27.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['linear_velocity'][-13:, 0], 'r')
	ax[1].plot(data[1200]['linear_velocity'][-13:, 1], 'r')
	ax[2].plot(data[1200]['linear_velocity'][-13:, 2], 'r')
	# ax[3].plot(data[1200]['linear_velocity'][-13:, 46], 'r')

	# plt.figure()
	# fft_0 = np.fft.fft(data[1200]['linear_velocity'][-13:, 2].flatten())
	# plt.plot(np.abs(fft_0))
	# plt.show()

	with open('data/train/speedway/2022-03-29-18-56-30.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['linear_velocity'][-13:, 0], 'k')
	ax[1].plot(data[1200]['linear_velocity'][-13:, 1], 'k')
	ax[2].plot(data[1200]['linear_velocity'][-13:, 2], 'k')
	# ax[3].plot(data[1200]['linear_velocity'][-14:, 46], 'k')

	plt.legend(['cement', 'grass', 'smallrocks', 'speedway'])

	plt.show()


	# create 4x1 subplots
	fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
	ax[0].plot(data[1200]['depth_info'][-13:, 0], 'b')
	ax[1].plot(data[1200]['depth_info'][-13:, 1], 'b')
	ax[2].plot(data[1200]['depth_info'][-13:, 2], 'b')
	ax[3].plot(data[1200]['depth_info'][-13:, 3], 'b')

	with open('data/train/grass/2022-03-14-18-45-10.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['depth_info'][-13:, 0], 'g')
	ax[1].plot(data[1200]['depth_info'][-13:, 1], 'g')
	ax[2].plot(data[1200]['depth_info'][-13:, 2], 'g')
	ax[3].plot(data[1200]['depth_info'][-13:, 3], 'g')

	with open('data/train/smallrocks/2022-03-15-10-15-27.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['depth_info'][-13:, 0], 'r')
	ax[1].plot(data[1200]['depth_info'][-13:, 1], 'r')
	ax[2].plot(data[1200]['depth_info'][-13:, 2], 'r')
	ax[3].plot(data[1200]['depth_info'][-13:, 3], 'r')

	with open('data/train/speedway/2022-03-29-18-56-30.pkl', 'rb') as f:
		data = pickle.load(f)

	# create 4x1 subplots
	ax[0].plot(data[1200]['depth_info'][-13:, 0], 'k')
	ax[1].plot(data[1200]['depth_info'][-13:, 1], 'k')
	ax[2].plot(data[1200]['depth_info'][-13:, 2], 'k')
	ax[3].plot(data[1200]['depth_info'][-14:, 3], 'k')

	plt.legend(['cement', 'grass', 'smallrocks', 'speedway'])

	plt.show()

