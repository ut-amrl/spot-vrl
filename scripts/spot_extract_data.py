#!/usr/bin/env python3

"""
A python file that reads the rosbag data and extracts time-synchronized images, odom, IMU data
and stores the data in a pickle file for easy access.
"""

import rosbag
import numpy as np
import pickle
import os
import argparse
import rospy
import cv2
from scipy.spatial.transform import Rotation as R
import random

from termcolor import cprint

from tqdm import tqdm

BEV_CAMERA_IMAGE_TOPIC = '/bev/single/compressed'
VECTORNAV_IMU_TOPIC = '/vectornav/IMU'
ODOM_TOPIC = '/odom'
SPOT_FEET_STATUS_TOPIC = '/spot/status/feet'
SPOT_LEG_STATUS_TOPIC = '/joint_states'

PATCH_SIZE = 64
PATCH_EPSILON = 0.5 * PATCH_SIZE * PATCH_SIZE

TOTAL_SAMPLES = 1000

class RosbagParser:
    def __init__(self, bag_file, output_dir_path, verbose=False):
        self.bag_file = bag_file
        self.output_dir_path = output_dir_path
        # find name of the bag file:
        bag_file_name = bag_file.split('/')[-1].split('.')[0]
        self.output_dir_path = os.path.join(self.output_dir_path, bag_file_name)
        cprint("Output directory: {}".format(self.output_dir_path), 'green')
        
        self.bag = rosbag.Bag(bag_file)
        
        # find total time in the rosbag in seconds
        self.total_time = self.find_total_time(self.bag)
        self.time_between_datapoints = self.total_time / TOTAL_SAMPLES
        
        print('Total time travelled: {} sec'.format(self.total_time))
        print('Min time between each data point: {} sec'.format(self.time_between_datapoints))        
        
        # find rate of IMU topic from rosbag
        # find number of messages in IMU topic
    
        # if verbose:
        #     print("Finding publish rate of IMU topic")
        #     self.imu_rate = int(self.find_msg_publish_rate(VECTORNAV_IMU_TOPIC))
        #     print("IMU rate: {} hz".format(self.imu_rate))
        
        self.feet_topic_rate = int(self.find_msg_publish_rate(SPOT_FEET_STATUS_TOPIC))
        self.leg_topic_rate = int(self.find_msg_publish_rate(SPOT_LEG_STATUS_TOPIC))
        cprint("Feet topic rate: {} hz".format(self.feet_topic_rate), 'yellow')
        cprint("Leg topic rate: {} hz".format(self.leg_topic_rate), 'yellow')
        
        self.verbose = verbose
        
        self.data = {}
        self.data['patches'], self.data['odom'], self.data['imu'] = [], [], []
        self.data['feet'], self.data['leg'] = [], []
        
        # print all the topics in the bag file
        cprint("Topics in the rosbag file: {}".format(bag_file), 'yellow')
        cprint("{}".format(self.bag.get_type_and_topic_info()[1].keys()), 'yellow')
        
        cprint("Parsing rosbag file: {}".format(bag_file), 'yellow')
        self.parse_bag(verbose)
        cprint("Done parsing rosbag file!!!", 'yellow')
        
    def parse_bag(self, verbose):
        """
        Parse the rosbag file and extract the data
        """
        self.camera_t, self.vectornav_t, self.odom_t = [], [], []
        self.feet_t, self.leg_t = [], []
        for topic, msg, t in self.bag.read_messages(topics=[BEV_CAMERA_IMAGE_TOPIC, 
                                                            VECTORNAV_IMU_TOPIC, 
                                                            ODOM_TOPIC,
                                                            SPOT_FEET_STATUS_TOPIC,
                                                            SPOT_LEG_STATUS_TOPIC]):
            if topic == BEV_CAMERA_IMAGE_TOPIC:
                if len(self.camera_t) == 0: self.camera_start_t = t
                self.camera_t.append(t)
            elif topic == VECTORNAV_IMU_TOPIC:
                if len(self.vectornav_t) == 0: self.vectornav_start_t = t
                self.vectornav_t.append(t)
            elif topic == ODOM_TOPIC:
                if len(self.odom_t) == 0: self.odom_start_t = t
                self.odom_t.append(t)
            elif topic == SPOT_FEET_STATUS_TOPIC:
                if len(self.feet_t) == 0: self.feet_start_t = t
                self.feet_t.append(t)
            elif topic == SPOT_LEG_STATUS_TOPIC:
                if len(self.leg_t) == 0: self.leg_start_t = t
                self.leg_t.append(t)
                
        # storage buffer to hold the past recent 50 BEV images which
        # we will use to extract patches
        self.storage_buffer = {'image': [], 'odom': []}
        
        for i, current_t in enumerate(tqdm(self.camera_t)):
            if verbose:
                print("Current time: {}".format(current_t.to_sec()-self.camera_start_t.to_sec()))
            
            # find the closest timestamp in the other topics
            closest_vectornav_t = self.find_closest_timestamp(self.vectornav_t, current_t)
            closest_odom_t = self.find_closest_timestamp(self.odom_t, current_t)
            closest_feet_t = self.find_closest_timestamp(self.feet_t, current_t)
            closest_leg_t = self.find_closest_timestamp(self.leg_t, current_t)
            
            if verbose:
                print('closest camera time: ', current_t.to_sec()-self.camera_start_t.to_sec())
                print('closest vectornav time: ', closest_vectornav_t.to_sec()-self.vectornav_start_t.to_sec())
                print('closest odom time: ', closest_odom_t.to_sec()-self.odom_start_t.to_sec())
                print('closest feet time: ', closest_feet_t.to_sec()-self.feet_start_t.to_sec())
                print('closest leg time: ', closest_leg_t.to_sec()-self.leg_start_t.to_sec())
                
            # wait for 10 seconds
            # if current_t.to_sec()-self.camera_start_t.to_sec() < 10: continue
            
            """
            IMU data
            """
            
            # find vector nav time from 2 seconds back
            vectornav_msg_generator = self.bag.read_messages(topics=[VECTORNAV_IMU_TOPIC], 
                                                             start_time=rospy.Time.from_sec(current_t.to_sec() - 2), 
                                                             end_time=closest_vectornav_t)
            # iterate through the generator and append the messages to the list
            imu_data = []
            for msg in vectornav_msg_generator: imu_data.append(msg[1])
            imu_data = imu_data[-400:] # get the last 400 messages
            
            if len(imu_data) != 400: 
                cprint('IMU data not enough', 'red')
                continue
            
            # convert IMU msg data to numpy array
            for ix in range(len(imu_data)):
                imu_data[ix] = np.array([imu_data[ix].angular_velocity.x, imu_data[ix].angular_velocity.y, imu_data[ix].angular_velocity.z,
                                        imu_data[ix].linear_acceleration.x, imu_data[ix].linear_acceleration.y, imu_data[ix].linear_acceleration.z,
                                        imu_data[ix].orientation.x, imu_data[ix].orientation.y, imu_data[ix].orientation.z, imu_data[ix].orientation.w])
            imu_data = np.array(imu_data)
            
            """
            FEET data
            """
            # find feet time from 2 seconds back
            feet_msg_generator = self.bag.read_messages(topics=[SPOT_FEET_STATUS_TOPIC], 
                                                             start_time=rospy.Time.from_sec(current_t.to_sec() - 2), 
                                                             end_time=closest_feet_t)
            # iterate through the generator and append the messages to the list
            feet_data = []
            for msg in feet_msg_generator: feet_data.append(msg[1])
            feet_data = feet_data[-int(self.feet_topic_rate * 2):] # get the last 2 sec worth of messages
            
            if len(feet_data) != int(self.feet_topic_rate * 2): 
                cprint('FEET data not enough', 'red')
                continue
            
            # convert feet msg data to numpy array
            for ix in range(len(feet_data)):
                feet_data[ix] = np.array([feet_data[ix].states[0].foot_position_rt_body.x, feet_data[ix].states[0].foot_position_rt_body.y, feet_data[ix].states[0].foot_position_rt_body.z, feet_data[ix].states[0].contact,
                                          feet_data[ix].states[0].visual_surface_ground_penetration_mean, feet_data[ix].states[0].visual_surface_ground_penetration_std,
                                          feet_data[ix].states[1].foot_position_rt_body.x, feet_data[ix].states[1].foot_position_rt_body.y, feet_data[ix].states[1].foot_position_rt_body.z, feet_data[ix].states[1].contact,
                                          feet_data[ix].states[1].visual_surface_ground_penetration_mean, feet_data[ix].states[1].visual_surface_ground_penetration_std,
                                          feet_data[ix].states[2].foot_position_rt_body.x, feet_data[ix].states[2].foot_position_rt_body.y, feet_data[ix].states[2].foot_position_rt_body.z, feet_data[ix].states[2].contact,
                                          feet_data[ix].states[2].visual_surface_ground_penetration_mean, feet_data[ix].states[2].visual_surface_ground_penetration_std,
                                          feet_data[ix].states[3].foot_position_rt_body.x, feet_data[ix].states[3].foot_position_rt_body.y, feet_data[ix].states[3].foot_position_rt_body.z, feet_data[ix].states[3].contact,
                                          feet_data[ix].states[3].visual_surface_ground_penetration_mean, feet_data[ix].states[3].visual_surface_ground_penetration_std])
            feet_data = np.array(feet_data)
                
            """
            LEG data
            """
            # find leg time from 2 seconds back
            leg_msg_generator = self.bag.read_messages(topics=[SPOT_LEG_STATUS_TOPIC], 
                                                             start_time=rospy.Time.from_sec(current_t.to_sec() - 2), 
                                                             end_time=closest_leg_t)
            # iterate through the generator and append the messages to the list
            leg_data = []
            for msg in leg_msg_generator: leg_data.append(msg[1])
            leg_data = leg_data[-int(self.leg_topic_rate * 2):] # get the last 2 sec worth of messages
            
            if len(leg_data) != int(self.leg_topic_rate * 2):
                cprint('LEG data not enough', 'red')
                continue
            
            # convert leg msg data to numpy array
            for ix in range(len(leg_data)):
                leg_data[ix] = np.concatenate((leg_data[ix].position, leg_data[ix].velocity, leg_data[ix].effort))
            leg_data = np.array(leg_data)
            
            """
            ODOM data
            """
            odom_data = next(self.bag.read_messages(topics=[ODOM_TOPIC], start_time=closest_odom_t, end_time=closest_odom_t))[1]
            
            # convert odom data from ROS msg to numpy array
            odom_data = np.array([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y, odom_data.pose.pose.position.z,
                                  odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w])
            
            """
            PATCH data
            """
            # store the data in the dictionary
            curr_bev_img = next(self.bag.read_messages(topics=[BEV_CAMERA_IMAGE_TOPIC], start_time=current_t, end_time=current_t))[1]
            
            # convert compressed image to numpy array
            curr_bev_img = np.fromstring(curr_bev_img.data, np.uint8)
            curr_bev_img = cv2.imdecode(curr_bev_img, cv2.IMREAD_COLOR)
            
            dist_moved = self.dist_between_odoms(odom_data, self.storage_buffer['odom'][-1]) if len(self.storage_buffer['odom']) > 0 else 0
            
            if len(self.storage_buffer['image']) == 0:
                self.storage_buffer['image'].append(curr_bev_img)
                self.storage_buffer['odom'].append(odom_data)
                # self.last_sample_dist = odom_data
                self.last_sample_time = current_t.to_sec()
            elif dist_moved > 0.1:
                self.storage_buffer['image'].append(curr_bev_img)
                self.storage_buffer['odom'].append(odom_data)
            
            # if self.dist_between_odoms(odom_data, self.last_sample_dist) < self.distance_between_datapoints: continue
            # else: self.last_sample_dist = odom_data
            
            if current_t.to_sec() - self.last_sample_time < self.time_between_datapoints: continue
            else: self.last_sample_time = current_t.to_sec()
                            
            patch_list = []
            
            # need to do the patch extraction here
            for j in range(len(self.storage_buffer['odom'])-1, -1, -1):
                prev_image = self.storage_buffer['image'][j]
                prev_odom = self.storage_buffer['odom'][j]
                
                # extract the patch
                patch, vis_img = self.extract_patch(odom_data, prev_odom, prev_image, visualize=True)
                
                if patch is not None:
                    patch_list.append(patch)
                    # save vis_img to file along with i nd j indices
                    # cv2.imwrite('images/vis/no_patch_{}_{}_{}.png'.format(i, current_t.to_sec()-self.camera_start_t.to_sec(), j), vis_img)
                    # print('Patch extracted and saved the vis image')
                
                # else:
                    # print('No patch was extracted')
                    
                if len(patch_list) == 10: break

            
            if len(patch_list) == 0:
                cprint('No patch was extracted', 'red')
            
            while len(self.storage_buffer['image']) > 15:
                self.storage_buffer['image'].pop(0)
                self.storage_buffer['odom'].pop(0)
                
            # if len(patch_list) > 0:
            #     print('Number of patches: ', len(patch_list))
            
            """
            APPEND ALL THE DATA TO self.data
            """
            self.data['patches'].append(patch_list)
            self.data['imu'].append(imu_data)
            self.data['odom'].append(odom_data)
            self.data['leg'].append(leg_data)
            self.data['feet'].append(feet_data)
            
            
        return self.data
    
    @staticmethod
    def dist_between_odoms(odom1, odom2):
        return np.linalg.norm(odom1[:2] - odom2[:2])
    
    def extract_patch(self, curr_pos, prev_pos, prev_image, visualize=False):
        curr_pos_np = np.array([curr_pos[0], curr_pos[1], 1])
        prev_pos_transform = np.zeros((3,3))
        prev_orientation = R.from_quat([prev_pos[3], prev_pos[4], prev_pos[5], prev_pos[6]]).as_euler('XYZ')[-1]
        prev_pos_transform[:2, :2] = R.from_euler('XYZ', [0, 0, prev_orientation]).as_matrix()[:2,:2] # figure this out
        prev_pos_transform[:, 2] = np.array([prev_pos[0], prev_pos[1], 1]).reshape((3))
        
        inv_pos_transform = np.linalg.inv(prev_pos_transform)
        curr_z_rotation = R.from_quat([curr_pos[3], curr_pos[4], curr_pos[5], curr_pos[6]]).as_euler('XYZ')[-1]
        curr_z_rotation = R.from_euler('XYZ', [0, 0, curr_z_rotation]).as_matrix()
        
        patch_corners = [
            curr_pos_np + curr_z_rotation @ np.array([0.5, 0.5, 0]),
            curr_pos_np + curr_z_rotation @ np.array([0.5, -0.5, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.5, -0.5, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.5, 0.5, 0])
        ]
        
        patch_corners_prev_frame = [
            inv_pos_transform @ patch_corners[0],
            inv_pos_transform @ patch_corners[1],
            inv_pos_transform @ patch_corners[2],
            inv_pos_transform @ patch_corners[3]
        ]
        
        scaled_patch_corners = [
            (patch_corners_prev_frame[0] * 150).astype(np.int),
            (patch_corners_prev_frame[1] * 150).astype(np.int),
            (patch_corners_prev_frame[2] * 150).astype(np.int),
            (patch_corners_prev_frame[3] * 150).astype(np.int),
        ]
        
        CENTER = np.array((1476//2, 749//2 + 320))
        patch_corners_image_frame = [
            CENTER + np.array((-scaled_patch_corners[0][1], -scaled_patch_corners[0][0])),
            CENTER + np.array((-scaled_patch_corners[1][1], -scaled_patch_corners[1][0])),
            CENTER + np.array((-scaled_patch_corners[2][1], -scaled_patch_corners[2][0])),
            CENTER + np.array((-scaled_patch_corners[3][1], -scaled_patch_corners[3][0]))
        ]
        
        vis_img = None
        if visualize:
            vis_img = prev_image.copy()

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
        
        patch = cv2.warpPerspective(prev_image, persp, (64, 64))
        
        # count number of black pixels in patch
        zero_count = np.count_nonzero(np.asarray(patch) == 0)
        if zero_count > PATCH_EPSILON: return None, vis_img
        
        return patch, vis_img
               
    def save_data(self):
        """
        Save the data to a pickle file
        """
        
        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)
        
        # find num of data points
        valid_id = []
        for ix in range(len(self.data['patches'])):
            if len(self.data['patches'][ix]) < 2: continue
            valid_id.append(ix)
        cprint("Num of valid data points: {}".format(len(valid_id)), 'green')
        
        print('the output dir path is: ', self.output_dir_path)
        
        for ix, valid_ix in enumerate(tqdm(valid_id)):
            # save this data point as a separate pickle file
            datapt = {'patches': self.data['patches'][valid_ix], 'imu': self.data['imu'][valid_ix], 'odom': self.data['odom'][valid_ix],
                      'leg': self.data['leg'][valid_ix], 'feet': self.data['feet'][valid_ix]}
            
            pickle_path = os.path.join(self.output_dir_path, "{}.pkl".format(ix))
            
            # save the data
            with open(pickle_path, 'wb') as f:
                pickle.dump(datapt, f)
        
    @staticmethod
    def find_total_distance_travelled(bag):
        # read the odom topic and find the total distance travelled
        dist = 0
        last_pos = None
        for _, msg, _ in bag.read_messages(topics=[ODOM_TOPIC]):
            if last_pos is None: 
                last_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                continue
            
            # find distance between last_pos and current pos
            current_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            dist += np.linalg.norm(current_pos - last_pos)
            last_pos = current_pos
        return dist
    
    @staticmethod
    def find_total_time(bag):
        # find the total time of the bag in seconds
        start_time = None
        end_time = None
        for _, _, t in bag.read_messages(topics=[ODOM_TOPIC]):
            if start_time is None: start_time = t
            end_time = t
        return (end_time - start_time).to_sec()
    
    @staticmethod
    def find_closest_timestamp(timestamps, current_t):
        """
        Find the closest timestamp in the list of timestamps using binary search
        """  
        left, right = 0, len(timestamps) - 1
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < current_t:
                left = mid + 1
            else:
                right = mid
        return timestamps[left]
        
    def find_msg_publish_rate(self, topic):
        """
        Find the publish rate of a topic
        """
        num_imu_msgs, imu_start_time, imu_end_time = 0, None, None
        for _, _, t in self.bag.read_messages(topics=[topic]):
            num_imu_msgs += 1
            if imu_start_time is None: imu_start_time = t
            imu_end_time = t
        return num_imu_msgs / (imu_end_time - imu_start_time).to_sec()
    
if __name__ == '__main__':
    # argparse
    args = argparse.ArgumentParser()
    args.add_argument('--bag_file', '-b', type=str, default='data/spot.bag')
    args.add_argument('--output_dir_path', '-o', type=str, default='spot_data/')
    args.add_argument('--verbose', '-v', action='store_true')
    args = args.parse_args()
    
    rosbag_parser = RosbagParser(args.bag_file, args.output_dir_path, verbose=args.verbose)
    
    cprint("saving data", 'green')
    rosbag_parser.save_data()
    
    cprint("*** DONE ***", 'green', attrs=['bold'])
            


