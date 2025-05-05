#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import mediapipe as mp


class MiRoClient:
    """
    Face tracking + head control + robot motion to follow a face using MiRo.
    """

    TICK = 0.02
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    DEBUG = True

    # Head limits
    TILT = 0.0
    LIFT_MIN = 0.1
    LIFT_MAX = 1.0
    YAW_LIMIT = 0.9
    PITCH_MIN = -0.35
    PITCH_MAX = 0.1

    # Motion control thresholds
    ANGULAR_SPEED = 0.15  # Rotation speed
    FORWARD_SPEED = 0.2   # Forward speed

    def __init__(self):
        rospy.init_node("miro_face_tracker_full", anonymous=True)
        rospy.sleep(1.0)

        robot_name = os.getenv("MIRO_ROBOT_NAME", "miro")
        topic_base_name = "/" + robot_name

        self.bridge = CvBridge()

        # Camera
        self.sub_cam = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Publishers
        self.config_pub = rospy.Publisher(
            topic_base_name + "/control/configuration/pos",
            Float64MultiArray,
            queue_size=0,
        )

        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel",
            TwistStamped,
            queue_size=0,
        )

        self.input_camera = None
        self.new_frame = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print("MiRo face+motion tracking started.")

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error converting image: %s", str(e))

    def calculate_head_pose(self, landmarks):
        # Key facial points
        nose = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # Convert to pixel space
        def px(pt): return int(pt.x * self.FRAME_WIDTH), int(pt.y * self.FRAME_HEIGHT)

        nose_pt = px(nose)
        chin_pt = px(chin)
        left_pt = px(left_eye)
        right_pt = px(right_eye)

        dx = right_pt[0] - left_pt[0]
        dy = chin_pt[1] - nose_pt[1]

        yaw = np.arctan2(dx, self.FRAME_WIDTH * 0.5)
        pitch = -np.arctan2(dy, self.FRAME_HEIGHT * 0.5)

        return yaw, pitch, nose_pt

    def send_head_configuration(self, yaw, pitch):
        lift = np.interp(abs(pitch), [0.0, 0.5], [self.LIFT_MAX, self.LIFT_MIN])
        yaw = float(np.clip(yaw, -self.YAW_LIMIT, self.YAW_LIMIT))
        pitch = float(np.clip(pitch, self.PITCH_MIN, self.PITCH_MAX))
        lift = float(np.clip(lift, self.LIFT_MIN, self.LIFT_MAX))

        msg = Float64MultiArray()
        msg.data = [self.TILT, lift, yaw, pitch]

        rospy.loginfo(f"Head config -> YAW: {yaw:.2f}, PITCH: {pitch:.2f}, LIFT: {lift:.2f}")
        self.config_pub.publish(msg)

    def move_robot(self, linear, angular):
        msg = TwistStamped()
        msg.twist.linear.x = linear
        msg.twist.angular.z = angular
        self.vel_pub.publish(msg)

    def follow_face(self):
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()

                results = self.face_mesh.process(frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    yaw, pitch, nose_pt = self.calculate_head_pose(landmarks)

                    # Head control
                    self.send_head_configuration(yaw, pitch)

                    # Body movement: keep face centered
                    face_x = nose_pt[0]
                    error_x = face_x - self.FRAME_WIDTH // 2
                    center_threshold = 40  # pixels
                    angular = 0.0
                    linear = 0.0

                    if abs(error_x) > center_threshold:
                        angular = -self.ANGULAR_SPEED if error_x < 0 else self.ANGULAR_SPEED

                    # Estimate distance by eye spacing
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]
                    eye_dist = abs(right_eye.x - left_eye.x)

                    if eye_dist < 0.07:  # Face is far
                        linear = self.FORWARD_SPEED
                    else:
                        linear = 0.0

                    self.move_robot(linear, angular)

                    if self.DEBUG:
                        cv2.circle(frame, nose_pt, 5, (255, 0, 0), -1)
                        cv2.putText(
                            frame,
                            f"YAW: {yaw:.2f}, PITCH: {pitch:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("MiRo Live Feed", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                else:
                    rospy.loginfo("No face detected.")
                    self.move_robot(0.0, 0.0)  # Stop moving

            rospy.sleep(self.TICK)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        client = MiRoClient()
        client.follow_face()
    except rospy.ROSInterruptException:
        pass
