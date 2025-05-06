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
    DEBUG = True  # Set True for OpenCV debug view
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

        self.pose_lock_threshold = 0.05
        self.success_counter_stage1 = 0
        self.success_counter_stage2 = 0
        self.stage_mode = 1
        self.face_last_seen = time.time()

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
            self.face_last_seen = time.time()
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

            if self.DEBUG:
                for lm in [nose_tip, left_eye, right_eye]:
                    cx = int(lm.x * self.FRAME_WIDTH)
                    cy = int(lm.y * self.FRAME_HEIGHT)
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

                cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return yaw, pitch, face_cx, frame

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

    def stage_one(self):
        rospy.loginfo("Stage 1: MiRo mimics the user's pose")
        while not rospy.is_shutdown():
            if time.time() - self.face_last_seen > 10:
                rospy.logwarn("No face detected. Switching to shutdown.")
                self.shutdown()
                return

            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()
                yaw, pitch, face_x, debug_frame = self.detect_head_pose(frame)

                if self.DEBUG:
                    cv2.imshow("MiRo Debug", debug_frame)
                    cv2.waitKey(1)

                if yaw is not None and pitch is not None:
                    if self.initial_yaw is None:
                        self.initial_yaw = yaw
                        self.initial_pitch = pitch
                        rospy.loginfo("Reference pose set.")
                        continue

                    rel_yaw = -(yaw - self.initial_yaw)
                    rel_pitch = pitch - self.initial_pitch

                    if (
                        abs(rel_yaw - self.prev_yaw) < self.pose_lock_threshold and
                        abs(rel_pitch - self.prev_pitch) < self.pose_lock_threshold
                    ):
                        continue

                    self.set_move_kinematic(yaw=rel_yaw, pitch=rel_pitch)
                    self.set_body_turn(face_x) if face_x else 0.0

                    self.prev_yaw = rel_yaw
                    self.prev_pitch = rel_pitch
                    self.success_counter_stage1 += 1

                    if self.success_counter_stage1 >= 10:
                        rospy.loginfo("Switching to Stage 2")
                        self.stage_mode = 2
                        return

            rospy.sleep(self.TICK)

    def stage_two(self):
        rospy.loginfo("Stage 2: User mimics MiRo's pose")
        while not rospy.is_shutdown():
            if time.time() - self.face_last_seen > 5:
                rospy.logwarn("No face detected. Shutting down.")
                self.shutdown()
                return

            target_yaw = random.choice([-0.4, 0.4])
            target_pitch = random.choice([-0.15, 0.1])
            rospy.loginfo(f"Pose set: yaw={target_yaw:.2f}, pitch={target_pitch:.2f}")
            self.set_move_kinematic(yaw=target_yaw, pitch=target_pitch)

            rospy.sleep(5.0)  # Pose hold before checking

            mimic_success = False
            wait_time = 0
            max_wait = 10.0  # Allow more time for user mimic

            while wait_time < max_wait:
                if self.new_frame:
                    self.new_frame = False
                    frame = self.input_camera.copy()
                    user_yaw, user_pitch, _, debug_frame = self.detect_head_pose(frame)

                    if self.DEBUG:
                        cv2.imshow("MiRo Debug", debug_frame)
                        cv2.waitKey(1)

                    if user_yaw is None or user_pitch is None:
                        wait_time += self.TICK
                        rospy.sleep(self.TICK)
                        continue

                    rel_yaw = -(user_yaw - self.initial_yaw)
                    rel_pitch = user_pitch - self.initial_pitch

                    yaw_diff = abs(rel_yaw - target_yaw)
                    pitch_diff = abs(rel_pitch - target_pitch)

                    if yaw_diff < self.pose_lock_threshold * 3 and pitch_diff < self.pose_lock_threshold * 3:
                        mimic_success = True
                        break

                wait_time += self.TICK
                rospy.sleep(self.TICK)

            if mimic_success:
                rospy.loginfo("User mimicked the pose!")
                self.success_counter_stage2 += 1
            else:
                rospy.loginfo("User did not mimic the pose.")

            if self.success_counter_stage2 >= 5:
                rospy.loginfo("Stage 2 complete!")
                self.shutdown()
                return

    def shutdown(self):
        if self.DEBUG:
            cv2.destroyAllWindows()
        rospy.loginfo("Shutting down. Stage 1: %d, Stage 2: %d",
                      self.success_counter_stage1, self.success_counter_stage2)
        rospy.signal_shutdown("Done")

    def run(self):
        rospy.loginfo("Waiting for face to initialize reference...")
        try:
            while not rospy.is_shutdown():
                if self.stage_mode == 1:
                    self.stage_one()
                else:
                    self.stage_two()
        except Exception as e:
            rospy.logerr("Fatal error: %s", str(e))
            self.shutdown()


if __name__ == "__main__":
    node = ImitateHeadPose()
    node.run()
