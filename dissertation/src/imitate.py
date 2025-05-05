#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import TwistStamped
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
        rospy.init_node("face_follower", anonymous=True)
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

        self.vel_pub = rospy.Publisher(
            topic_base + "/control/cmd_vel", TwistStamped, queue_size=0
        )

        self.joint_pub = rospy.Publisher(
            topic_base + "/control/kinematic_joints", JointState, queue_size=0
        )

        self.input_camera = None
        self.new_frame = False

        # For pose estimation
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

    def detect_face_and_pose(self, frame):
        face_pos = None
        yaw = pitch = None
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            rospy.loginfo_once("Face detected.")
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark
            nose_tip = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            eye_diff_x = right_eye.x - left_eye.x
            yaw = np.clip(eye_diff_x * 10.0, -0.9, 0.9)

            eye_center_y = (left_eye.y + right_eye.y) / 2
            pitch = np.clip((nose_tip.y - eye_center_y) * 10.0, -0.35, 0.1)

            h, w, _ = frame.shape
            cx = int(nose_tip.x * w)
            cy = int(nose_tip.y * h)
            face_pos = (cx, cy)
        else:
            rospy.logwarn_once("No face detected.")

        return face_pos, yaw, pitch

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

    def follow_face(self):
        print("MiRo is following a face. Press CTRL+C to stop.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()

                face_position, est_yaw, est_pitch = self.detect_face_and_pose(frame)
                if face_position is not None:
                    cx, cy = face_position
                    error_x = cx - self.FRAME_WIDTH // 2
                    error_y = cy - self.FRAME_HEIGHT // 2

                    # Head adjustments
                    new_yaw = self.current_yaw + (error_x / self.FRAME_WIDTH) * 0.05
                    new_lift = self.current_lift - (error_y / self.FRAME_HEIGHT) * 0.05
                    new_pitch = est_pitch if est_pitch is not None else self.current_pitch
                    self.set_head_joints(lift=new_lift, yaw=new_yaw, pitch=new_pitch)

                    # Movement control
                    forward_speed = -error_y / self.FRAME_HEIGHT * 0.2  # Tune gain
                    angular_speed = -error_x / self.FRAME_WIDTH * 0.5   # Tune gain

                    move_cmd = TwistStamped()
                    move_cmd.twist.linear.x = np.clip(forward_speed, -0.1, 0.1)
                    move_cmd.twist.angular.z = np.clip(angular_speed, -0.3, 0.3)
                    self.vel_pub.publish(move_cmd)

                    if self.DEBUG:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.imshow("MiRo Face View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                else:
                    # Scan slowly
                    self.set_head_joints(yaw=self.current_yaw + 0.02)
                    stop_cmd = TwistStamped()
                    self.vel_pub.publish(stop_cmd)

            rospy.sleep(self.TICK)

if __name__ == "__main__":
    try:
        main = MiRoClient()
        main.follow_face()
    except rospy.ROSInterruptException:
        pass
