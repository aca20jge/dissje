#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import mediapipe as mp


class MiRoClient:
    """
    Script to track a face, map head pose (yaw + pitch) to MiRo's head config,
    and send configuration commands directly to control the robot's kinematic chain.
    """

    TICK = 0.02  # Control loop frequency (50 Hz)
    YAW_LIMIT = 1.0  # Max yaw in radians (~57 deg)
    PITCH_LIMIT = 0.5  # Max pitch in radians (~28 deg)
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    DEBUG = True

    def __init__(self):
        rospy.init_node("miro_head_pose_follower", anonymous=True)
        rospy.sleep(1.0)

        # Get robot name
        robot_name = os.getenv("MIRO_ROBOT_NAME", "miro")
        topic_base_name = "/" + robot_name

        self.bridge = CvBridge()

        # Camera input
        self.sub_cam = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Configuration publisher
        self.config_pub = rospy.Publisher(
            topic_base_name + "/control/configuration/pos",
            Float64MultiArray,
            queue_size=0,
        )

        self.input_camera = None
        self.new_frame = False

        # MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print("MiRo head tracking node started. Press CTRL+C to stop.")

    def callback_cam(self, ros_image):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error converting camera image: %s", str(e))

    def calculate_head_pose(self, landmarks, image_width, image_height):
        """
        Estimate head yaw and pitch from selected facial landmarks.
        """
        # Key points: nose tip (1), chin (152), left eye (33), right eye (263)
        nose = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        # Convert to pixel coordinates
        def to_pixel_coords(pt):
            return int(pt.x * image_width), int(pt.y * image_height)

        nose_pt = to_pixel_coords(nose)
        chin_pt = to_pixel_coords(chin)
        left_pt = to_pixel_coords(left_eye)
        right_pt = to_pixel_coords(right_eye)

        # Calculate horizontal and vertical angles
        dx = right_pt[0] - left_pt[0]
        dy = chin_pt[1] - nose_pt[1]

        yaw = np.arctan2(dx, image_width * 0.5)
        pitch = -np.arctan2(dy, image_height * 0.5)

        return yaw, pitch

    def send_head_configuration(self, yaw, pitch):
        """
        Send a full head configuration: [TILT, LIFT, YAW, PITCH]
        """
        TILT = 0.0
        LIFT = 0.0

        yaw = float(np.clip(yaw, -self.YAW_LIMIT, self.YAW_LIMIT))
        pitch = float(np.clip(pitch, -self.PITCH_LIMIT, self.PITCH_LIMIT))

        msg = Float64MultiArray()
        msg.data = [TILT, LIFT, yaw, pitch]
        rospy.loginfo(f"Head config -> YAW: {yaw:.2f} rad, PITCH: {pitch:.2f} rad")

        self.config_pub.publish(msg)

    def follow_face(self):
        """
        Main loop to detect face, estimate head orientation, and update MiRo's head.
        """
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera.copy()

                results = self.face_mesh.process(frame)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    yaw, pitch = self.calculate_head_pose(
                        landmarks, self.FRAME_WIDTH, self.FRAME_HEIGHT
                    )

                    self.send_head_configuration(yaw, pitch)

                    # Visual feedback
                    if self.DEBUG:
                        cv2.putText(
                            frame,
                            f"YAW: {yaw:.2f}, PITCH: {pitch:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
                        cv2.imshow("MiRo Camera - Face Tracking", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                else:
                    rospy.loginfo("No face detected.")

            rospy.sleep(self.TICK)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = MiRoClient()
        node.follow_face()
    except rospy.ROSInterruptException:
        pass
