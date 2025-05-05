#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import subprocess

stage2_started = False

def status_callback(msg):
    global stage2_started
    if msg.data == "stage1_complete" and not stage2_started:
        stage2_started = True
        subprocess.Popen(["rosrun", "miro_pose_imitation", "stage2_node.py"])

if __name__ == "__main__":
    rospy.init_node("main_controller")
    rospy.Subscriber("/stage_status", String, status_callback)
    subprocess.Popen(["rosrun", "miro_pose_imitation", "stage1_node.py"])
    rospy.spin()
