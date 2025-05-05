#!/usr/bin/env python3

import os
import time
import rospy
import cv2
import numpy as np
import random
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import UInt32MultiArray, String
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import mediapipe as mp
import miro2 as miro

class ImitateHeadPose:
    DEBUG = False
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    TICK = 0.2

    def __init__(self, stage=1):
        self.stage = stage
        self.bridge = CvBridge()
        topic_base = "/" + os.getenv("MIRO_ROBOT_NAME", "miro")

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

        self.illum_pub = rospy.Publisher(
            topic_base + "/control/illum", UInt32MultiArray, queue_size=1
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

        self.success_counter = 0
        self.stage2_poses = []
        self.last_face_seen = time.time()

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.input_camera = image
            self.new_frame = True
            self.last_face_seen = time.time()
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", str(e))

    def detect_head_pose(self, frame):
        yaw = pitch = None
        face_cx = None
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            eye_diff_x = right_eye.x - left_eye.x
            eye_center_y = (left_eye.y + right_eye.y) / 2

            yaw = eye_diff_x * 10.0
            pitch = (nose_tip.y - eye_center_y) * 10.0

            face_cx = int(nose_tip.x * self.FRAME_WIDTH)

        return yaw, pitch, face_cx

    def set_move_kinematic(self, yaw=0.0, pitch=0.0):
        joint_cmd = JointState()
        joint_cmd.position = [
            miro.constants.TILT_RAD_CALIB,
            miro.constants.LIFT_RAD_CALIB,
            np.clip(yaw, -0.9, 0.9),
            np.clip(pitch, -0.35, 0.1),
        ]
        self.kinematic_pub.publish(joint_cmd)

    def set_body_turn(self, face_x):
        center_x = self.FRAME_WIDTH // 2
        error_x = face_x - center_x
        deadzone = 30

        twist = TwistStamped()
        delta_turn = 0.0

        if abs(error_x) > deadzone:
            turn_speed = -float(error_x) / center_x * 0.2
            delta_turn = np.clip(turn_speed, -0.3, 0.3)
            twist.twist.angular.z = delta_turn

        self.vel_pub.publish(twist)
        return delta_turn

    def expressive_feedback(self):
        led_msg = UInt32MultiArray()
        colors = [
            [0, 179, 0], [0, 102, 77], [0, 0, 179],
            [77, 0, 128], [153, 0, 51], [0, 179, 0]
        ]
        for color in colors:
            led_msg.data = color * 6
            self.illum_pub.publish(led_msg)
            rospy.sleep(0.03)
        led_msg.data = [0, 0, 0] * 6
        self.illum_pub.publish(led_msg)

    def run(self):
        rospy.loginfo(f"Stage {self.stage} running")
        r = rospy.Rate(1.0 / self.TICK)

        if self.stage == 1:
            self.run_stage1(r)
        else:
            self.run_stage2(r)

    def run_stage1(self, r):
        while not rospy.is_shutdown() and self.success_counter < 10:
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()
                yaw, pitch, face_x = self.detect_head_pose(frame)

                if yaw is not None and pitch is not None:
                    if self.initial_yaw is None:
                        self.initial_yaw = yaw
                        self.initial_pitch = pitch
                        continue

                    rel_yaw = -(yaw - self.initial_yaw)
                    rel_pitch = pitch - self.initial_pitch

                    if (abs(rel_yaw - self.prev_yaw) < 0.05 and
                        abs(rel_pitch - self.prev_pitch) < 0.05):
                        self.success_counter += 1
                        self.expressive_feedback()
                        self.initial_yaw = None
                        self.initial_pitch = None
                        continue

                    self.set_move_kinematic(yaw=rel_yaw, pitch=rel_pitch)
                    if face_x is not None:
                        self.set_body_turn(face_x)

                    self.prev_yaw = rel_yaw
                    self.prev_pitch = rel_pitch
            r.sleep()

    def run_stage2(self, r):
        for i in range(20):
            if time.time() - self.last_face_seen > 5.0:
                rospy.logwarn("Face lost. Exiting stage 2.")
                print(f"Stage 1 Successes: 10")
                print(f"Stage 2 Successes: {self.success_counter}")
                return

            yaw = random.uniform(-0.5, 0.5)
            pitch = random.uniform(-0.2, 0.1)
            self.set_move_kinematic(yaw, pitch)
            rospy.sleep(2.0)

            success = False
            for _ in range(25):
                if self.new_frame:
                    self.new_frame = False
                    frame = self.input_camera.copy()
                    user_yaw, user_pitch, _ = self.detect_head_pose(frame)

                    if user_yaw is not None and user_pitch is not None:
                        rel_yaw = -(user_yaw - yaw)
                        rel_pitch = user_pitch - pitch
                        if abs(rel_yaw) < 0.05 and abs(rel_pitch) < 0.05:
                            success = True
                            break
                r.sleep()

            if success:
                self.success_counter += 1
                self.expressive_feedback()

        print(f"Stage 1 Successes: 10")
        print(f"Stage 2 Successes: {self.success_counter}")