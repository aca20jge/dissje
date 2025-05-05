#!/usr/bin/env python3

import os
import time
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import mediapipe as mp
import miro2 as miro


class ImitateHeadPose:
    DEBUG = True
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TICK = 0.02

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

        self.input_camera = None
        self.new_frame = False

        # Mediapipe FaceMesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.initial_yaw = None
        self.initial_pitch = None

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", str(e))

    def detect_head_pose(self, frame):
        yaw = pitch = None
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

        return yaw, pitch

    def set_move_kinematic(self, yaw=0.0, pitch=0.0):
        joint_cmd = JointState()
        joint_cmd.position = [
            miro.constants.TILT_RAD_CALIB,                     # tilt (not used)
            miro.constants.LIFT_RAD_CALIB,                     # lift (default)
            np.clip(yaw, -0.9, 0.9),                            # yaw
            np.clip(pitch, -0.35, 0.1),                         # pitch
        ]
        self.kinematic_pub.publish(joint_cmd)

    def imitate_head_pose(self):
        rospy.loginfo("Waiting to detect a face to set center...")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()
                yaw, pitch = self.detect_head_pose(frame)

                if yaw is not None and pitch is not None:
                    if self.initial_yaw is None:
                        self.initial_yaw = yaw
                        self.initial_pitch = pitch
                        rospy.loginfo("Face detected. Setting head center reference.")
                        continue

                    # Compute relative deltas
                    rel_yaw = yaw - self.initial_yaw
                    rel_pitch = pitch - self.initial_pitch

                    # Apply movement
                    self.set_move_kinematic(yaw=rel_yaw, pitch=rel_pitch)

                    if self.DEBUG:
                        cv2.putText(frame, f"Yaw Δ: {rel_yaw:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Pitch Δ: {rel_pitch:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("MiRo Imitate View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            rospy.sleep(self.TICK)


if __name__ == "__main__":
    node = ImitateHeadPose()
    node.imitate_head_pose()
