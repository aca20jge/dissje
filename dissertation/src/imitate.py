#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import os
import mediapipe as mp
import miro2 as miro  # MiRo SDK

class MiRoImitator:
    TICK = 0.02
    DEBUG = True

    def __init__(self):
        rospy.init_node("imitate_head_pose", anonymous=True)
        rospy.sleep(1.0)

        self.bridge = CvBridge()

        robot_name = os.getenv("MIRO_ROBOT_NAME")
        if not robot_name:
            rospy.logerr("MIRO_ROBOT_NAME not set!")
            raise ValueError("MIRO_ROBOT_NAME environment variable is required.")
        topic_base = "/" + robot_name

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

        self.cosmetic_pub = rospy.Publisher(
            topic_base + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.input_camera = None
        self.new_frame = False

    def callback_cam(self, ros_image):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            self.input_camera = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", str(e))

    def detect_head_pose(self, frame):
        yaw = pitch = None
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Estimate yaw from horizontal eye difference
            eye_diff_x = right_eye.x - left_eye.x
            yaw = np.clip(eye_diff_x * 10.0, -0.9, 0.9)

            # Estimate pitch from nose position relative to eyes
            eye_center_y = (left_eye.y + right_eye.y) / 2
            pitch = np.clip((nose_tip.y - eye_center_y) * 10.0, -0.35, 0.1)

        return yaw, pitch

    def set_move_kinematic(self, yaw, pitch):
        joint_cmd = JointState()

        tilt = miro.constants.TILT_RAD_CALIB  # fixed (not used)
        lift = miro.constants.LIFT_RAD_CALIB  # neutral lift

        joint_cmd.position = [tilt, lift, yaw, pitch]
        self.kinematic_pub.publish(joint_cmd)

    def imitate_head_pose(self):
        rospy.loginfo("MiRo is imitating your head pose.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()
                yaw, pitch = self.detect_head_pose(frame)

                if yaw is not None and pitch is not None:
                    self.set_move_kinematic(yaw, pitch)

                    if self.DEBUG:
                        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("MiRo Head View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            rospy.sleep(self.TICK)

if __name__ == "__main__":
    try:
        imitator = MiRoImitator()
        imitator.imitate_head_pose()
    except rospy.ROSInterruptException:
        pass
