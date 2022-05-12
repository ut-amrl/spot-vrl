#!/usr/bin/env python
"""
A rosnode that listens to camera and legoloam localization info
and saves the processed data into a pickle file.
"""

from copyreg import pickle
import numpy as np
import time
import rospy
import os
import cv2
import message_filters
import subprocess
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Imu
import tf2_ros
from termcolor import cprint

class ListenRecordData:
    """
        A class for a ROS node that listens to camera, IMU and legoloam localization info
        and saves the processed data into a pickle file after the rosbag play is finished.
        """

    def __init__(self, save_data_path, rosbag_play_process):
        self.rosbag_play_process = rosbag_play_process
        self.save_data_path = save_data_path

        # we have 2 imus - one from jackal and one from azure kinect
        self.imu_msgs_jackal = np.zeros((200, 3), dtype=np.float32)  # past 200 imu messages ~ 1.98 s
        self.imu_msgs_kinect = np.zeros((200, 3), dtype=np.float32)  # past 200 imu messages ~ 1.98 s

        # ros tf
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.msg_data = {
            'terrain_patch': [],
            'imu_kinect_history': [],
            'imu_jackal_history': [],
            'odom': []
        }

        # counter to keep count of the number of messages received
        self.counter = 0

        # subscribe to accel, gyro
        rospy.Subscriber('/imu/data', Imu, self.imu_callback_jackal)
        rospy.Subscriber('/kinect_imu', Imu, self.imu_callback_kinect)
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)

        # image = message_filters.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage)
        # imu = message_filters.Subscriber('/kinect_imu', Imu)
        # jackal_imu = message_filters.Subscriber('/imu/data', Imu)
        # ts = message_filters.TimeSynchronizer([image, imu, jackal_imu], 20, 0.1, allow_headerless=True)
        # ts.registerCallback(self.callback)

    def imu_callback_jackal(self, msg):
        self.imu_msgs_jackal = np.roll(self.imu_msgs_jackal, -1, axis=0)
        self.imu_msgs_jackal[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    def imu_callback_kinect(self, msg):
        self.imu_msgs_kinect = np.roll(self.imu_msgs_kinect, -1, axis=0)
        self.imu_msgs_kinect[-1] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                             msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    
    def get_localization(self):
        self.trans = self.tfBuffer.lookup_transform('/map', '/base_link', rospy.Time())
        print('trans : ', self.trans)

    def image_callback(self, image):
        print('Receiving data... : ', self.counter)

        self.msg_data['terrain_patch'].append(image)
        self.msg_data['imu_kinect_history'].append(self.imu_msgs_kinect.flatten())
        self.msg_data['imu_jackal_history'].append(self.imu_msgs_jackal.flatten())
        self.msg_data['odom'].append(self.trans)

        self.counter += 1

    def save_data(self):
        data = {}
        
        # IMU data
        data['imu_kinect_history'] = self.msg_data['imu_kinect_history']
        del self.msg_data['imu_kinect_history']
        data['imu_jackal_history'] = self.msg_data['imu_jackal_history']
        del self.msg_data['imu_jackal_history']
        
        # Vision data
        data['patches'] = self.process_bev_image_and_patches(self.msg_data)
        del self.msg_data
        
        # dumping data
        cprint('Saving data...{}'.format(len(data['imu_kinect_history'])), 'yellow')
        pickle.dump(data, open(os.path.join(self.save_data_path, 'data.pkl'), 'wb'))
        cprint('Saved data successfully ', 'yellow', attrs=['blink'])

    def get_patch_from_odom_delta(self, T_odom_curr, T_odom_prev, prevImage, visualize=False):
        T_curr_prev = np.linalg.inv(T_odom_curr) @ T_odom_prev
        T_prev_curr = np.linalg.inv(T_curr_prev)

        # patch corners in current robot frame
        height = T_odom_prev[2, -1]
        patch_corners = [
            np.array([0.5, 0.5, 0, 1]),
            np.array([0.5, -0.5, 0, 1]),
            np.array([-0.5, -0.5, 0, 1]),
            np.array([-0.5, 0.5, 0, 1])
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

        persp = cv2.getPerspectiveTransform(np.float32(patch_corners_image_frame), np.float32([[0, 0], [127, 0], [127, 127], [0, 127]]))
        patch = cv2.warpPerspective(
            prevImage,
            persp,
            (128, 128)
        )

        zero_count = (patch == 0)
        if np.sum(zero_count) > PATCH_EPSILON:
            return None, 1.0, None

        return patch, (np.sum(zero_count) / (PATCH_SIZE * PATCH_SIZE)), vis_img


if __name__ == '__main__':
    # create the node
    rospy.init_node('patch_extractor', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('save_data_path')
    
    print('rosbag_path: ', rosbag_path)
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROSBAG path does not exist')
    
    # start a subprocess to play the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1'])
    
    # start the rosbag recorder    
    recorder = ListenRecordData(save_data_path=save_data_path, rosbag_play_process=rosbag_play_process)
    
    while not rospy.is_shutdown():
        recorder.get_localization()
        if rosbag_play_process.poll() is not None:
            print('rosbag_play_process is finished')
            recorder.save_data()
            exit(0)
    
    rospy.spin()
    
    