#!/usr/bin/env python3.6
"""
A rosnode that listens to camera and legoloam localization info
and saves the processed data into a pickle file.
"""

# import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
from copyreg import pickle
import numpy as np
import time

from torch import dtype
import rospy
import os
import cv2
import message_filters
import subprocess
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Imu
import tf2_ros
from termcolor import cprint
from scipy.spatial.transform import Rotation as R

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

C_i = np.asarray([[983.322571,   0.,         1021.098450        ],
                    [  0.,         983.123108, 775.020630        ],
                    [  0.,           0.,           1.        ]]).reshape((3, 3))

C_i_inv = np.linalg.inv(C_i)

class ListenRecordData:
    """
        A class for a ROS node that listens to camera, IMU and legoloam localization info
        and saves the processed data into a pickle file after the rosbag play is finished.
        """

    def __init__(self, save_data_path, rosbag_play_process):
        self.rosbag_play_process = rosbag_play_process
        self.save_data_path = save_data_path

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
        rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, self.image_callback)

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

    def image_callback(self, image):
        print('Receiving data... : ', self.counter)
        
        if self.trans is None: return
        self.msg_data['image_msg'].append(image)
        self.msg_data['imu_kinect_history'].append(self.imu_msgs_kinect.flatten())
        self.msg_data['imu_jackal_history'].append(self.imu_msgs_jackal.flatten())
        self.msg_data['odom'].append(self.trans)
        self.msg_data['imu_jackal_orientation'].append(self.imu_jackal_orientation)
        
        
        bevimage, img = ListenRecordData.camera_imu_homography(self.msg_data['imu_jackal_orientation'][-1], self.msg_data['image_msg'][-1])
        img = cv2.resize(img, (bevimage.shape[1]//3, bevimage.shape[0]//3))
        bevimage = cv2.resize(bevimage, (bevimage.shape[1]//3, bevimage.shape[0]//3))
        cv2.imshow('disp', np.hstack((bevimage, img)))
        # cv2.imshow('2', img)
        
        cv2.waitKey(1)

        self.counter += 1

    def save_data(self):
        data = {} # dict to hold all the processed data
        
        # IMU data
        cprint('Processing inertial data...', 'green', attrs=['bold'])
        data['imu_kinect_history'] = self.msg_data['imu_kinect_history']
        del self.msg_data['imu_kinect_history']
        data['imu_jackal_history'] = self.msg_data['imu_jackal_history']
        del self.msg_data['imu_jackal_history']
        
        # Vision data
        cprint('Processing vision data...', 'green', attrs=['bold'])
        data['patches'] = self.process_bev_image_and_patches(self.msg_data)
        del self.msg_data['image_msg']
        
        # dumping data
        cprint('Saving data...{}'.format(len(data['imu_kinect_history'])), 'yellow')
        pickle.dump(data, open(os.path.join(self.save_data_path, 'data.pkl'), 'wb'))
        cprint('Saved data successfully ', 'yellow', attrs=['blink'])
        
    def process_bev_image_and_patches(self, msg_data):
        processed_data = {'image':[]}
        msg_data['patches'] = {}
        msg_data['patches_found'] = {}

        for i in tqdm(range(len(msg_data['image_msg']))):
            bevimage, _ = ListenRecordData.camera_imu_homography(msg_data['imu_jackal_orientation'][i], msg_data['image_msg'][i])
            processed_data['image'].append(bevimage)

            # cv2.imshow('disp', bevimage)
            # cv2.waitKey(1)

            #save this one image msg and the vector nav msg as a pickle file

            # pickle.dump(msg_data['image_msg'][i], open('/home/haresh/PycharmProjects/visual_IKD/tmp/image_msg.pkl', 'wb'))
            # pickle.dump(msg_data['vectornav'][i], open('/home/haresh/PycharmProjects/visual_IKD/tmp/imu_msg.pkl', 'wb'))
            # input()

            # now find the patches for this image
            curr_odom = msg_data['odom_msg'][i]

            found_patch = False
            for j in range(i, max(i-30, 0), -2):
                prev_image = processed_data['image'][j]
                prev_odom = msg_data['odom_msg'][j]
                # cv2.imshow('src_image', processed_data['src_image'][i])
                patch, patch_black_pct, curr_img, vis_img = ListenRecordData.get_patch_from_odom_delta(
                    curr_odom.pose.pose, prev_odom.pose.pose, curr_odom.twist.twist,
                    prev_odom.twist.twist, prev_image, processed_data['image'][i])
                if patch is not None:
                    found_patch = True
                    if i not in msg_data['patches']:
                        msg_data['patches'][i] = []
                    msg_data['patches'][i].append(patch)

                # stop adding more than 10 patches for a single data point
                if found_patch and len(msg_data['patches'][i]) > 5: break

            if not found_patch:
                print("Unable to find patch for idx: ", i)
                msg_data['patches'][i] = [processed_data['image'][i][500:564, 613:677]]

            # remove the i-30th image from RAM
            if i > 30:
                processed_data['image'][i-30] = None

            # was the patch found or no ?
            if found_patch: msg_data['patches_found'][i] = True
            else: msg_data['patches_found'][i] = False

        return msg_data['patches'], msg_data['patches_found']

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

    @staticmethod
    def camera_imu_homography(orientation_quat, image):

        # R_imu_world = R.from_quat(orientation_quat)
        # R_imu_world = R_imu_world.as_euler('XYZ', degrees=True)
        # R_imu_world[0] = 0.0
        # R_imu_world[1] = 0.0
        # R_imu_world[2] = 0.0

        # R_imu_world = R_imu_world
        # R_imu_world = R.from_euler('xyz', R_imu_world, degrees=True)
        

        R_cam_imu = R.from_euler("xyz", [-90, 90, 0], degrees=True)
        R_fix = R.from_euler("XYZ", [26.5, 0, 0], degrees=True)
        
        R1 =  R_fix * R_cam_imu  #* R_imu_world
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
        
        print('img shape : ', img.shape)
        

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
    
    print('rosbag_path: ', rosbag_path)
    if not os.path.exists(rosbag_path):
        raise FileNotFoundError('ROSBAG path does not exist')
    
    # start a subprocess to play the rosbag
    rosbag_play_process = subprocess.Popen(['rosbag', 'play', rosbag_path, '-r', '1','--clock'])
    
    # start the rosbag recorder    
    recorder = ListenRecordData(save_data_path=save_data_path, rosbag_play_process=rosbag_play_process)
    
    while not rospy.is_shutdown():
        recorder.get_localization()
        if rosbag_play_process.poll() is not None:
            print('rosbag_play_process is finished')
            recorder.save_data()
            exit(0)
    
    rospy.spin()
    rosbag_play_process.kill()
    
    