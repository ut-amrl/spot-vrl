import loguru
import numpy as np
from spot_vrl.data.synced_data import SynchronizedData
import cv2
import argparse
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import glob

PATCH_SIZE = 64
PATCH_EPSILON = 0.5 * PATCH_SIZE * PATCH_SIZE

def affineinverse(M):
	tmp = np.hstack((M[:3, :3].T, -M[:3, :3].T @ M[:3, 3].reshape((4, 1))))
	return np.vstack((tmp, np.array([0, 0, 0, 1])))

def get_patch_from_odom_delta(T_odom_curr, T_odom_prev, prevImage, visualize=False):
	T_curr_prev = np.linalg.inv(T_odom_curr) @ T_odom_prev
	T_prev_curr = np.linalg.inv(T_curr_prev)

	# patch corners in current robot frame
	height = T_odom_prev[2, -1]
	patch_corners = [
		np.array([0.3, 0.3, 0, 1]),
		np.array([0.3, -0.3, 0, 1]),
		np.array([-0.3, -0.3, 0, 1]),
		np.array([-0.3, 0.3, 0, 1])
	]

	# patch corners in prev frame
	patch_corners_prev_frame = [
		T_prev_curr @ patch_corners[0],
		T_prev_curr @ patch_corners[1],
		T_prev_curr @ patch_corners[2],
		T_prev_curr @ patch_corners[3],
	]

	scaled_patch_corners = [
		(patch_corners_prev_frame[0] * 150).astype(np.int),
		(patch_corners_prev_frame[1] * 150).astype(np.int),
		(patch_corners_prev_frame[2] * 150).astype(np.int),
		(patch_corners_prev_frame[3] * 150).astype(np.int),
	]

	# TODO : fix this number
	CENTER = np.array((394, 480))
	patch_corners_image_frame = [
		CENTER + np.array((-scaled_patch_corners[0][1], -scaled_patch_corners[0][0])),
		CENTER + np.array((-scaled_patch_corners[1][1], -scaled_patch_corners[1][0])),
		CENTER + np.array((-scaled_patch_corners[2][1], -scaled_patch_corners[2][0])),
		CENTER + np.array((-scaled_patch_corners[3][1], -scaled_patch_corners[3][0]))
	]

	vis_img = None
	if visualize:
		vis_img = prevImage.copy()

		# draw the patch rectangle
		cv2.line(
			vis_img,
			(patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
			(patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
			(0, 255, 0),
			2
		)
		cv2.line(
			vis_img,
			(patch_corners_image_frame[1][0], patch_corners_image_frame[1][1]),
			(patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
			(0, 255, 0),
			2
		)
		cv2.line(
			vis_img,
			(patch_corners_image_frame[2][0], patch_corners_image_frame[2][1]),
			(patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
			(0, 255, 0),
			2
		)
		cv2.line(
			vis_img,
			(patch_corners_image_frame[3][0], patch_corners_image_frame[3][1]),
			(patch_corners_image_frame[0][0], patch_corners_image_frame[0][1]),
			(0, 255, 0),
			2
		)

	persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame), np.float32([[0, 0], [63, 0], [63, 63], [0, 63]]))
	patch = cv2.warpPerspective(
		prevImage,
		persp,
		(64, 64)
	)

	zero_count = (patch == 0)
	if np.sum(zero_count) > PATCH_EPSILON:
		return None, 1.0, None

	return patch, (np.sum(zero_count) / (PATCH_SIZE * PATCH_SIZE)), vis_img


def process_collected_data(filename, visualize=False):
	terraindata = SynchronizedData(filename=filename)
	processedterraindata = []
	for i in tqdm(range(len(terraindata.data)-1, -1, -1)):
		datapt = {
			'patches': [],
			'inertial': terraindata.data[i].imu_history
		}

		# current image and odom
		currImage = terraindata.data[i].image
		currOdom = terraindata.data[i].odom

		# for this location underneath the robot, find the visual patch
		# from previous observations
		for j in range(i, max(i-150, 0), -2):
			prevImage = terraindata.data[j].image
			prevOdom = terraindata.data[j].odom
			patch, _, visImg = get_patch_from_odom_delta(currOdom, prevOdom, prevImage, visualize=visualize)

			if patch is not None:
				datapt['patches'].append(patch)
				if visualize:
					visImg = cv2.putText(visImg, 'i : '+str(i), (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
					visImg = cv2.putText(visImg, 'j : '+str(j), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
					cv2.imshow('visimg', visImg)
					cv2.imshow('patch', patch)
					cv2.waitKey(0)

			if len(datapt['patches']) > 20 : break

		if len(datapt['patches']) == 0: continue
		# add datapt to our dict only if there is atleast 1 patch
		processedterraindata.append(datapt)
	return processedterraindata


def process_and_save_as_pickle(filename, visualize=False):
	try:
		# store this in a pickle file for reading later
		processedterraindata = process_collected_data(filename=filename, visualize=visualize)
		pickle.dump(processedterraindata, open(filename.replace('bddf', 'pkl'), 'wb'))
		# also save a tiny dataset
		pickle.dump(processedterraindata[:20], open(filename.replace('.bddf', '_short.pkl'), 'wb'))
		print('Processed this data and saved as a pickle file..')
	except Exception as e:
		print(e)

if __name__ == '__main__':
	# processedterraindata = process_and_save_as_pickle(filename='data/2022-03-14-17-36-52.bddf')
	bddf_file_paths = glob.glob('data/*/*/*.bddf', recursive=True)
	assert len(bddf_file_paths) > 0, "No file found. Check if the files exist / file path is correctly assigned"
	for bddf_file in bddf_file_paths:
		print('Processing BDDF file : ', bddf_file)
		processedterraindata = process_and_save_as_pickle(filename=bddf_file, visualize=False)

	print('Done..')


