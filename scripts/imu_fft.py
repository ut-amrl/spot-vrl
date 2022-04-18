import numpy as np
import matplotlib.pyplot as plt
from spot_vrl.data.synced_data import SynchronizedData
from scipy import fftpack

grass_imu_info = np.asarray(SynchronizedData(filename='data_tmp/train_cache/grass/2022-03-14-18-45-10.bddf').imu_container.all_sensor_data)
speedway_imu_info = np.asarray(SynchronizedData(filename='data_tmp/train_cache/speedway/2022-03-29-18-56-30.bddf').imu_container.all_sensor_data)
cement_imu_info = np.asarray(SynchronizedData(filename='data_tmp/train/cement/2022-03-14-17-36-52.bddf').imu_container.all_sensor_data)
smallrocks_imu_info = np.asarray(SynchronizedData(filename='data_tmp/train/smallrocks/2022-03-15-10-15-27.bddf').imu_container.all_sensor_data)

linear_velocity_grass = grass_imu_info[:, 37:40]
linear_velocity_speedway = speedway_imu_info[:, 37:40]
linear_velocity_cement = cement_imu_info[:, 37:40]
linear_velocity_smallrocks = smallrocks_imu_info[:, 37:40]

angular_velocity_grass = grass_imu_info[:, 40:43]


# create 3 subplots
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
axs[0].plot(np.abs(fftpack.fft(linear_velocity_grass[300:, 0]).real)[1:121], label='grass')
axs[0].plot(np.abs(fftpack.fft(linear_velocity_speedway[300:, 0]).real)[1:121], label='speedway')
axs[0].plot(np.abs(fftpack.fft(linear_velocity_cement[300:, 0]).real)[1:121], label='cement')
axs[0].plot(np.abs(fftpack.fft(linear_velocity_smallrocks[300:, 0]).real)[1:121], label='smallrocks')
axs[1].plot(np.abs(fftpack.fft(linear_velocity_grass[300:, 1]).real)[1:121], label='grass')
axs[1].plot(np.abs(fftpack.fft(linear_velocity_speedway[300:, 1]).real)[1:121], label='speedway')
axs[1].plot(np.abs(fftpack.fft(linear_velocity_cement[300:, 1]).real)[1:121], label='cement')
axs[1].plot(np.abs(fftpack.fft(linear_velocity_smallrocks[300:, 1]).real)[1:121], label='smallrocks')
axs[2].plot(np.abs(fftpack.fft(linear_velocity_grass[300:, 2]).real)[1:121], label='grass')
axs[2].plot(np.abs(fftpack.fft(linear_velocity_speedway[300:, 2]).real)[1:121], label='speedway')
axs[2].plot(np.abs(fftpack.fft(linear_velocity_cement[300:, 2]).real)[1:121], label='cement')
axs[2].plot(np.abs(fftpack.fft(linear_velocity_smallrocks[300:, 2]).real)[1:121], label='smallrocks')
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
axs[0].plot(linear_velocity_grass[300:500, 0], label='grass')
axs[0].plot(linear_velocity_speedway[300:500, 0], label='speedway')
axs[0].plot(linear_velocity_cement[300:500, 0], label='cement')
axs[0].plot(linear_velocity_smallrocks[300:500, 0], label='smallrocks')
axs[1].plot(linear_velocity_grass[300:500, 1], label='grass')
axs[1].plot(linear_velocity_speedway[300:500, 1], label='speedway')
axs[1].plot(linear_velocity_cement[300:500, 1], label='cement')
axs[1].plot(linear_velocity_smallrocks[300:500, 1], label='smallrocks')
axs[2].plot(linear_velocity_grass[300:500, 2], label='grass')
axs[2].plot(linear_velocity_speedway[300:500, 2], label='speedway')
axs[2].plot(linear_velocity_cement[300:500, 2], label='cement')
axs[2].plot(linear_velocity_smallrocks[300:500, 2], label='smallrocks')
plt.legend()
plt.show()




