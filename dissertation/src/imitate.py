#!/usr/bin/env python3

import os
import time
import random
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import UInt32MultiArray
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge

import mediapipe as mp
import miro2 as miro


class ImitateHeadPose:
    DEBUG = True
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    TICK = 0.3

    def __init__(self):
        rospy.init_node("imitate_head_pose", anonymous=True)
        rospy.sleep(1.0)

        self.bridge = CvBridge()

        topic_base = "/" + os.getenv("MIRO_ROBOT_NAME")

        self.sub_cam = rospy.Subscriber(
            topic_base + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.kinematic_pub = rospy.Publisher(
            topic_base + "/control/kinematic_joints", JointState, queue_size=0
        )

        self.vel_pub = rospy.Publisher(
            topic_base + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        self.input_camera = None
        self.new_frame = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.initial_yaw = None
        self.initial_pitch = None

        self.prev_yaw = 0.0
        self.prev_pitch = 0.0
        self.prev_turn = 0.0

        self.pose_lock_threshold = 0.3
