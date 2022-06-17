#!/usr/bin/env python3.6
"""
A rosnode that listens to camera and legoloam localization info
and saves the processed data into a pickle file.
"""

# import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
# from copyreg import pickle
import pickle
import numpy as np
import time
from torch import dtype
from tqdm import tqdm
import rospy
import os
import cv2
import message_filters
import subprocess
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Imu
import tf2_ros
from termcolor import cprint
import message_filters
from scipy.spatial.transform import Rotation as R

PATCH_SIZE = 64
PATCH_EPSILON = 0.5 * PATCH_SIZE * PATCH_SIZE
ACTUATION_LATENCY = 0.25

# camera matrix for the azure kinect
# C_i = np.array(
#     [622.0649233612024, 0.0, 633.1717569157071, 0.0, 619.7990184421728, 368.0688607187958, 0.0, 0.0, 1.0]).reshape(
#     (3, 3))


# camera parameters of the jackal
# C_i = np.array([623.3087 ,   0.     , 636.79787,
#            0.     , 624.91863, 366.72814,
#            0.     ,   0.     ,   1.     ]).reshape((3, 3))


# C_i = np.asarray([[935.30743609,   0.,         960.        ],
#                     [  0.,         935.30743609, 540.        ],
#                     [  0.,           0.,           1.        ]]).reshape((3, 3))

# C_i = np.asarray([[983.322571,   0.,         1021.098450        ],
#                     [  0.,         983.123108, 775.020630        ],
#                     [  0.,           0.,           1.        ]]).reshape((3, 3))

C_i = np.asarray([[983.322571,   0.,         1024        ],
                    [  0.,         983.123108, 768        ],
                    [  0.,           0.,           1.        ]]).reshape((3, 3))

C_i_inv = np.linalg.inv(C_i)

class ListenRecordData:
    """
        A class for a ROS node that listens to camera, IMU and legoloam localization info
        and saves the processed data into a pickle file after the rosbag play is finished.
        """

    def __init__(self, save_data_path, rosbag_play_process, visualize_results=False):
        self.rosbag_play_process = rosbag_play_process
        self.save_data_path = save_data_path
        self.visualize_results = visualize_results

        # we have 2 imus - one from jackal and one from azure kinect
        self.imu_msgs_jackal = np.zeros((200, 6), dtype=np.float32)  # past 200 imu messages ~ 1.98 s
        self.imu_msgs_kinect = np.zeros((200, 6), dtype=np.float32)  # past 200 imu messages ~ 1.98 s

        # ros tf
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.msg_data = {
            'image_msg': [],
            'imu_kinect_history': [],
            'imu_jackal_history': [],
            'imu_jackal_orientation': [],
            'odom': []
        }

        # counter to keep count of the number of messages received
        self.counter = 0

        # subscribe to accel, gyro
        rospy.Subscriber('/imu/data', Imu, self.imu_callback_jackal)
        rospy.Subscriber('/kinect_imu', Imu, self.imu_callback_kinect)
        # rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)
        
        # approximate syncronization of the odom and camera topics
        image_topic = message_filters.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage)
        odom_topic = message_filters.Subscriber('/jackal_velocity_controller/odom', Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([image_topic, odom_topic], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.image_odom_callback)
        
    def imu_callback_jackal(self, msg):
        self.imu_msgs_jackal = np.roll(self.imu_msgs_jackal, -1, axis=0)
        self.imu_msgs_jackal[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_jackal_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def imu_callback_kinect(self, msg):
        self.imu_msgs_kinect = np.roll(self.imu_msgs_kinect, -1, axis=0)
        self.imu_msgs_kinect[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    
    def get_localization(self):
        try:
            self.trans = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time(), rospy.Duration(1.0))
        except Exception as e:
            self.trans = None
            print(str(e))

    def image_odom_callback(self, image, odom):
        print('Receiving data... : ', self.counter)
        
        orientation = R.from_quat([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]).as_euler('XYZ')[-1]
        odom_val = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y, orientation])
        
        self.msg_data['image_msg'].append(image)
        self.msg_data['imu_kinect_history'].append(self.imu_msgs_kinect.flatten())
        self.msg_data['imu_jackal_history'].append(self.imu_msgs_jackal.flatten())
        self.msg_data['odom'].append(odom_val.copy())
        self.msg_data['imu_jackal_orientation'].append(self.imu_jackal_orientation)

    def save_data(self):
        # dict to hold all the processed data
        data = {
            'patches': [],
            'imu_kinect': [],
            'imu_jackal': []
        } 
        
        # storage buffer to hold the past recent 20 BEV images on which we perform patch extraction
        self.storage_buffer = {'image':[], 'odom':[]}
        
        for i in tqdm(range(len(self.msg_data['image_msg']))):
            bevimage, _ = ListenRecordData.camera_imu_homography(self.msg_data['imu_jackal_orientation'][i], self.msg_data['image_msg'][i])
            self.storage_buffer['image'].append(bevimage)
            self.storage_buffer['odom'].append(self.msg_data['odom'][i])
            
            if self.visualize_results:
                bevimage = cv2.resize(bevimage, (bevimage.shape[1]//3, bevimage.shape[0]//3))
            
            # now find the patches for this image
            curr_odom = self.msg_data['odom'][i]
            
            patch_list = []
            
            # search for the patches in the storage buffer
            for j in range(0, len(self.storage_buffer['odom'])):
                # print('j : ', j)
                prev_image = self.storage_buffer['image'][j]
                prev_odom = self.storage_buffer['odom'][j]
                
                # extract the patch
                patch, vis_img = ListenRecordData.get_patch_from_odom_delta(curr_odom, 
                                                                            prev_odom,
                                                                            prev_image, 
                                                                            visualize=self.visualize_results)
                if patch is not None:
                    patch_list.append(patch)
                    
                    if self.visualize_results:
                        vis_img = cv2.resize(vis_img, (vis_img.shape[1]//3, vis_img.shape[0]//3))
                        cv2.imshow('current img <-> previous img', np.hstack((bevimage, vis_img)))
                        cv2.waitKey(5)
                    
                if len(patch_list) >= 10: break
                
            while len(self.storage_buffer['image']) > 20:
                self.storage_buffer['image'].pop(0)
                self.storage_buffer['odom'].pop(0)
                
            if len(patch_list) > 0:
                print('Num patches : ', len(patch_list))
                # patches have been found. add it to data dict
                data['patches'].append(patch_list)
                data['imu_jackal'].append(self.msg_data['imu_jackal_history'][i])
                data['imu_kinect'].append(self.msg_data['imu_kinect_history'][i])
                                
        # dumping data
        cprint('Saving data of size {}'.format(len(data['imu_kinect'])), 'yellow')
        cprint('Keys in the dataset : '+str(data.keys()), 'yellow')
        pickle.dump(data, open(os.path.join(self.save_data_path, 'data.pkl'), 'wb'))
        cprint('Saved data successfully ', 'yellow', attrs=['blink'])

    @staticmethod
    def get_patch_from_odom_delta(curr_pos, prev_pos, prev_image, visualize=False):
        curr_pos_np = np.array([curr_pos[0], curr_pos[1], 1])
        prev_pos_transform = np.zeros((3, 3))
        prev_pos_transform[:2, :2] = R.from_euler('XYZ', [0, 0, prev_pos[2]]).as_matrix()[:2,:2] # figure this out
        prev_pos_transform[:, 2] = np.array([prev_pos[0], prev_pos[1], 1]).reshape((3))

        inv_pos_transform = np.linalg.inv(prev_pos_transform)
        curr_z_rotation = R.from_euler('XYZ', [0, 0, curr_pos[2]]).as_matrix()

        patch_corners = [
            curr_pos_np + curr_z_rotation @ np.array([0.3, 0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([0.3, -0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.3, -0.3, 0]),
            curr_pos_np + curr_z_rotation @ np.array([-0.3, 0.3, 0])
        ]
        patch_corners_prev_frame = [
            inv_pos_transform @ patch_corners[0],
            inv_pos_transform @ patch_corners[1],
            inv_pos_transform @ patch_corners[2],
            inv_pos_transform @ patch_corners[3],
        ]
        scaled_patch_corners = [
            (patch_corners_prev_frame[0] * 132.003788).astype(np.int),
            (patch_corners_prev_frame[1] * 132.003788).astype(np.int),
            (patch_corners_prev_frame[2] * 132.003788).astype(np.int),
            (patch_corners_prev_frame[3] * 132.003788).astype(np.int),
        ]
        
        CENTER = np.array((1024-20, (768-55)*2))
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

        patch = cv2.warpPerspective(
            prev_image,
            persp,
            (64, 64)
        )

        zero_count = np.logical_and(np.logical_and(patch[:, :, 0] == 0, patch[:, :, 1] == 0), patch[:, :, 2] == 0)

        if np.sum(zero_count) > PATCH_EPSILON:
            return None, vis_img

        return patch, vis_img
    
    @staticmethod
    def camera_imu_homography(orientation_quat, image):

        R_imu_world = R.from_quat(orientation_quat)
        R_imu_world = R_imu_world.as_euler('XYZ', degrees=True)

        
        R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
        # R_pitch = R.from_euler("XYZ", [26.5+R_imu_world[0], 0, R_imu_world[1]], degrees=True)
        R_pitch = R.from_euler("XYZ", [26.5, 0, 0], degrees=True)
        
        R1 = R_pitch * R_cam_imu
        R1 = R1.as_matrix()
        t1 = R1 @ np.array([0., 0., 0.75]).reshape((3, 1))
        
        R2 = R.from_euler("XYZ", [-180, 0, 90], degrees=True).as_matrix()
        t2 = R2 @ np.array([4.20, 0., 6.0]).reshape((3, 1))
        
        n = np.array([0, 0, 1]).reshape((3, 1))
        n1 = R1 @ n

        H12 = ListenRecordData.homography_camera_displacement(R1, R2, t1, t2, n1)
        homography_matrix = C_i @ H12 @ C_i_inv
        homography_matrix /= homography_matrix[2, 2]
                
        img = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        output = cv2.warpPerspective(img, homography_matrix,  (img.shape[1], img.shape[0]))
        
        # flip output horizontally
        output = cv2.flip(output, 1)

        return output, img.copy()

    @staticmethod
    def homography_camera_displacement(R1, R2, t1, t2, n1):
        R12 = R2 @ R1.T
        t12 = R2 @ (- R1.T @ t1) + t2
        # d is distance from plane to t1.
        d = np.linalg.norm(n1.dot(t1.T))

        H12 = R12 + ((t12 @ n1.T) / d)
        H12 /= H12[2, 2]
        return H12
    
if __name__ == '__main__':
    # create the node
    rospy.init_node('patch_extractor', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('save_data_path')
    visualize_results = rospy.get_param('visualize_results')
    print('visualize results bool : ', visualize_results)
    
    print('rosbag_path: ', rosbag_path)
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROSBAG path does not exist')
    
    # start a subprocess to play the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1','--clock'])
    
    # start the rosbag recorder    
    recorder = ListenRecordData(save_data_path=save_data_path, 
                                rosbag_play_process=rosbag_play_process,
                                visualize_results=visualize_results)
    
    while not rospy.is_shutdown():
        recorder.get_localization()
        if rosbag_play_process.poll() is not None:
            print('rosbag_play_process is finished, now saving the extracted patch data into a pickle file...')
            recorder.save_data()
            exit(0)
    
    rospy.spin()
    rosbag_play_process.kill()
    
    