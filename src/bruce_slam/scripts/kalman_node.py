#!/usr/bin/env python

# Python 导入
import rospy

# 导入卡尔曼滤波代码
from bruce_slam.utils.io import *
from bruce_slam.kalman import KalmanNode


if __name__ == "__main__":
    rospy.init_node("kalman", log_level=rospy.INFO)

    node = KalmanNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("开始在线卡尔曼滤波...")
        rospy.spin()
    else:
        loginfo("开始离线卡尔曼滤波...")
        offline(args)
