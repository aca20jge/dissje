#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from imitate_head_pose import ImitateHeadPose

if __name__ == "__main__":
    rospy.init_node("stage1_node")
    node = ImitateHeadPose(stage=1)
    node.run()
    pub = rospy.Publisher("/stage_status", String, queue_size=1, latch=True)
    rospy.sleep(1.0)
    pub.publish("stage1_complete")
