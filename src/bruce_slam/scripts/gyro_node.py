#!/usr/bin/env python
import rospy
from bruce_slam.utils.io import *
from bruce_slam.gyro import GyroFilter

if __name__ == "__main__":
    rospy.init_node("gyro_fusion", log_level=rospy.INFO)

    node = GyroFilter()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("开始陀螺仪融合...")
        rospy.spin()
    else:
        loginfo("开始陀螺仪融合...")
