#!/usr/bin/env python3

import os
import time
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge

import mediapipe as mp
import miro2 as miro


class ImitateHeadPose:
    DEBUG = False
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    TICK = 0.5

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

        self.illum_pub = rospy.Publisher(
            topic_base + "/control/illum", Float32MultiArray, queue_size=0
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

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.input_camera = image
            self.new_frame = True
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
        led_msg = Float32MultiArray()
        sequence = [
            [0.0, 0.7, 0.0],   # green
            [0.0, 0.4, 0.3],   # teal
            [0.0, 0.0, 0.7],   # blue
            [0.3, 0.0, 0.5],   # purple
            [0.6, 0.0, 0.2],   # pink
            [0.0, 0.7, 0.0],   # back to green
        ]
        for color in sequence:
            led_msg.data = color * 6
            self.illum_pub.publish(led_msg)
            rospy.sleep(0.05)
        led_msg.data = [0.0, 0.0, 0.0] * 6
        self.illum_pub.publish(led_msg)

    def imitate_head_pose(self):
        rospy.loginfo("Waiting for face to initialize reference...")

        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()

                yaw, pitch, face_x = self.detect_head_pose(frame)

                if yaw is not None and pitch is not None:
                    if self.initial_yaw is None:
                        self.initial_yaw = yaw
                        self.initial_pitch = pitch
                        rospy.loginfo("Reference pose set.")
                        continue

                    rel_yaw = -(yaw - self.initial_yaw)  # flipped yaw direction
                    rel_pitch = pitch - self.initial_pitch

                    delta_yaw = abs(rel_yaw - self.prev_yaw)
                    delta_pitch = abs(rel_pitch - self.prev_pitch)

                    self.set_move_kinematic(yaw=rel_yaw, pitch=rel_pitch)

                    delta_turn = 0
                    if face_x is not None:
                        delta_turn = abs(self.set_body_turn(face_x) - self.prev_turn)

                    if delta_yaw > 0.05 or delta_pitch > 0.05 or delta_turn > 0.05:
                        self.expressive_feedback()

                    self.prev_yaw = rel_yaw
                    self.prev_pitch = rel_pitch
                    self.prev_turn = delta_turn

                    if self.DEBUG:
                        cv2.putText(frame, f"Yaw Δ: {rel_yaw:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Pitch Δ: {rel_pitch:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("MiRo Debug View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            rospy.sleep(self.TICK)


if __name__ == "__main__":
    node = ImitateHeadPose()
    node.imitate_head_pose()
