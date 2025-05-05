#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage, JointState
from cv_bridge import CvBridge
import numpy as np
import os
import mediapipe as mp

class MiRoClient:
    TICK = 0.02
    DEBUG = True
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    def __init__(self):
        rospy.init_node("imitate_face_pose", anonymous=True)
        rospy.sleep(2.0)
        self.image_converter = CvBridge()

        topic_base = "/" + os.getenv("MIRO_ROBOT_NAME")
        self.sub_cam = rospy.Subscriber(
            topic_base + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.joint_pub = rospy.Publisher(
            topic_base + "/control/kinematic_joints", JointState, queue_size=0
        )

        self.input_camera = None
        self.new_frame = False

        # MediaPipe for face mesh and pose estimation
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.current_lift = 0.5
        self.current_yaw = 0.0
        self.current_pitch = 0.0

    def callback_cam(self, ros_image):
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error converting image: %s", str(e))

    def detect_head_pose(self, frame):
        yaw = pitch = None
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Estimate yaw (side-to-side)
            eye_diff_x = right_eye.x - left_eye.x
            yaw = np.clip(eye_diff_x * 10.0, -0.9, 0.9)

            # Estimate pitch (up-down)
            eye_center_y = (left_eye.y + right_eye.y) / 2
            pitch = np.clip((nose_tip.y - eye_center_y) * 10.0, -0.35, 0.1)

        return yaw, pitch

    def set_head_joints(self, lift=None, yaw=None, pitch=None):
        joint_msg = JointState()
        joint_msg.name = []
        joint_msg.position = []

        if lift is not None:
            self.current_lift = np.clip(lift, 0.1, 1.0)
            joint_msg.name.append("head_lift_joint")
            joint_msg.position.append(self.current_lift)

        if yaw is not None:
            self.current_yaw = np.clip(yaw, -0.9, 0.9)
            joint_msg.name.append("head_yaw_joint")
            joint_msg.position.append(self.current_yaw)

        if pitch is not None:
            self.current_pitch = np.clip(pitch, -0.35, 0.1)
            joint_msg.name.append("head_pitch_joint")
            joint_msg.position.append(self.current_pitch)

        self.joint_pub.publish(joint_msg)

    def imitate_head_pose(self):
        print("MiRo is mimicking your head pose. Press CTRL+C to stop.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()

                yaw, pitch = self.detect_head_pose(frame)
                if yaw is not None and pitch is not None:
                    self.set_head_joints(yaw=yaw, pitch=pitch)

                    if self.DEBUG:
                        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("MiRo Imitation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                else:
                    rospy.logwarn_once("Face detected but pose could not be estimated.")

            rospy.sleep(self.TICK)

if __name__ == "__main__":
    try:
        main = MiRoClient()
        main.imitate_head_pose()
    except rospy.ROSInterruptException:
        pass
