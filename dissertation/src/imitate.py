#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import numpy as np
import os
import mediapipe as mp

class MiRoClient:
    TICK = 0.5  # Update interval for the control loop
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    YAW_LIMIT = 0.5  # Yaw limit in radians
    PITCH_LIMIT = 0.5  # Pitch limit in radians

    def __init__(self):
        rospy.init_node("face_mimic", anonymous=True)
        rospy.sleep(2.0)
        self.image_converter = CvBridge()

        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME", "miro")
        self.sub_cam = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_cam,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.head_yaw_pub = rospy.Publisher(
            topic_base_name + "/control/head_yaw/pos", Float64, queue_size=0
        )
        self.head_pitch_pub = rospy.Publisher(
            topic_base_name + "/control/head_pitch/pos", Float64, queue_size=0
        )

        self.input_camera = None
        self.new_frame = False

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

        # Camera internals
        self.camera_matrix = np.array([
            [self.FRAME_WIDTH, 0, self.FRAME_WIDTH / 2],
            [0, self.FRAME_WIDTH, self.FRAME_HEIGHT / 2],
            [0, 0, 1]
        ], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))

        # 3D model points for pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -63.6, -12.5),      # Chin
            (-43.3, 32.7, -26.0),     # Left eye left corner
            (43.3, 32.7, -26.0),      # Right eye right corner
            (-28.9, -28.9, -24.1),    # Left mouth corner
            (28.9, -28.9, -24.1)      # Right mouth corner
        ])
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def callback_cam(self, ros_image):
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            self.input_camera = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.new_frame = True
        except Exception as e:
            rospy.logerr("Error in converting image: %s", str(e))

    def mimic_face(self):
        print("MiRo is mimicking head movements and centering face. Press CTRL+C to stop.")
        while not rospy.is_shutdown():
            if self.new_frame:
                self.new_frame = False
                frame = self.input_camera
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    # ----- Pose Estimation Using solvePnP -----
                    image_points = []
                    for idx in self.landmark_indices:
                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * self.FRAME_WIDTH)
                        y = int(lm.y * self.FRAME_HEIGHT)
                        image_points.append((x, y))

                    image_points = np.array(image_points, dtype="double")

                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        self.model_points, image_points, self.camera_matrix, self.dist_coeffs
                    )

                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                        # Calculate yaw and pitch from rotation matrix
                        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0],
                                                    np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)))
                        pitch = np.degrees(np.arctan2(rotation_matrix[0, 2],
                                                      rotation_matrix[2, 2]))

                        yaw_radians = np.radians(yaw)
                        pitch_radians = np.radians(pitch)

                        # ----- Face Centering Based on Nose Tip -----
                        nose_tip = face_landmarks.landmark[1]
                        x = int(nose_tip.x * self.FRAME_WIDTH)
                        y = int(nose_tip.y * self.FRAME_HEIGHT)

                        dx = x - self.FRAME_WIDTH // 2
                        dy = y - self.FRAME_HEIGHT // 2

                        norm_dx = dx / (self.FRAME_WIDTH / 2)
                        norm_dy = dy / (self.FRAME_HEIGHT / 2)

                        correction_yaw = -norm_dx * 0.1  # small correction factor
                        correction_pitch = -norm_dy * 0.1

                        # Combine pose estimation and centering correction
                        combined_yaw = np.clip(yaw_radians + correction_yaw, -self.YAW_LIMIT, self.YAW_LIMIT)
                        combined_pitch = np.clip(pitch_radians + correction_pitch, -self.PITCH_LIMIT, self.PITCH_LIMIT)

                        # Debug
                        print(f"[Mimic] Yaw: {yaw_radians:.2f}, Pitch: {pitch_radians:.2f}")
                        print(f"[Centering] dx: {dx}, dy: {dy}, Correction: yaw {correction_yaw:.2f}, pitch {correction_pitch:.2f}")
                        print(f"[Combined] Final Yaw: {combined_yaw:.2f}, Final Pitch: {combined_pitch:.2f}")

                        # Publish to MiRo
                        self.head_yaw_pub.publish(Float64(combined_yaw))
                        self.head_pitch_pub.publish(Float64(combined_pitch))

                    # Draw landmarks for visualization
                    for landmark in face_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # Display the image
                cv2.imshow("MiRo Camera Feed", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rospy.sleep(self.TICK)

        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == "__main__":
    node = MiRoClient()
    node.mimic_face()
