#!/usr/bin/env python

# Python 导入
import rospy

# 导入航迹推算代码
from bruce_slam.utils.io import *
from bruce_slam.dead_reckoning import DeadReckoningNode


if __name__ == "__main__":
    rospy.init_node("localization", log_level=rospy.INFO)

    node = DeadReckoningNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("开始在线定位...")
        rospy.spin()
    else:
        loginfo("开始离线定位...")
        offline(args)
