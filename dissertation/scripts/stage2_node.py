#!/usr/bin/env python3

import rospy
from imitate_head_pose import ImitateHeadPose

if __name__ == "__main__":
    rospy.init_node("stage2_node")
    node = ImitateHeadPose(stage=2)
    node.run()
