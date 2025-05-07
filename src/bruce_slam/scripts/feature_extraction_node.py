#!/usr/bin/env python
import rospy
from bruce_slam.utils.io import *
from bruce_slam.feature_extraction import FeatureExtraction

if __name__ == "__main__":

    # 初始化 ROS 节点
    rospy.init_node("feature_extraction_node", log_level=rospy.INFO)

    # 调用类构造函数
    node = FeatureExtraction()
    node.init_node()

    # 获取参数
    parser = common_parser()
    args, _ = parser.parse_known_args()

    # 记录日志并运行
    if not args.file:
        loginfo("开始在线声纳特征提取...")
        rospy.spin()
    else:
        loginfo("开始离线声纳特征提取...")
        offline(args)
