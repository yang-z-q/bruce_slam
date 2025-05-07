#!/usr/bin/env python

import rospy
from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.topics import *

def offline(args)->None:
    """离线运行 SLAM 系统

    参数:
        args (Any): 运行系统所需的参数
    """

    # 导入所需的额外模块
    from rosgraph_msgs.msg import Clock
    from dead_reckoning_node import DeadReckoningNode
    from feature_extraction_node import FeatureExtraction
    from gyro_node import GyroFilter
    from mapping_node import MappingNode
    from bruce_slam.utils import io

    # 设置一些参数
    io.offline = True
    node.save_fig = False
    node.save_data = False

    # 实例化所需的节点
    dead_reckoning_node = DeadReckoningNode()
    dead_reckoning_node.init_node(SLAM_NS + "localization/")
    feature_extraction_node = FeatureExtraction()
    feature_extraction_node.init_node(SLAM_NS + "feature_extraction/")
    gyro_node = GyroFilter()
    gyro_node.init_node(SLAM_NS + "gyro/")
    """mp_node = MappingNode()
    mp_node.init_node(SLAM_NS + "mapping/")"""
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)

    # 遍历整个 rosbag
    # for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
    #     while not rospy.is_shutdown():
    #         if callback_lock_event.wait(1.0):
    #             break

    #     if rospy.is_shutdown():
    #         break

    #     if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
    #         dead_reckoning_node.imu_sub.callback(msg)
    #     elif topic == DVL_TOPIC:
    #         dead_reckoning_node.dvl_sub.callback(msg)
    #     elif topic == DEPTH_TOPIC:
    #         dead_reckoning_node.depth_sub.callback(msg)
    #     elif topic == SONAR_TOPIC or SONAR_TOPIC_UNCOMPRESSED:
    #         feature_extraction_node.sonar_sub.callback(msg)
    #     elif topic == GYRO_TOPIC:
    #         gyro_node.gyro_sub.callback(msg)

    #     # 使用 IMU 驱动时钟
    #     if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:

    #         clock_pub.publish(Clock(msg.header.stamp))

    #         # 发布 map 到 world 的变换，以便在 rviz 中以 z-down 坐标系可视化所有内容
    #         node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")
    output_txt_path = "bag_messages.txt"
    with open(output_txt_path, 'w') as txt_file:
        start_time = None
        # 遍历整个 rosbag
        for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
            while not rospy.is_shutdown():
                if callback_lock_event.wait(1.0):
                    break

            if rospy.is_shutdown():
                break

            # 记录起始时间
            if start_time is None:
                start_time = msg.header.stamp.to_sec()

            # 计算当前时间与起始时间的差值
            elapsed_time = msg.header.stamp.to_sec() - start_time
            if elapsed_time > 10:
                break

            # 将话题和消息内容写入 TXT 文件
            txt_file.write(f"Topic: {topic}\n")
            txt_file.write(f"Message: {msg}\n")
            txt_file.write("-" * 80 + "\n")

            if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
                dead_reckoning_node.imu_sub.callback(msg)
            elif topic == DVL_TOPIC:
                dead_reckoning_node.dvl_sub.callback(msg)
            elif topic == DEPTH_TOPIC:
                dead_reckoning_node.depth_sub.callback(msg)
            elif topic == SONAR_TOPIC or SONAR_TOPIC_UNCOMPRESSED:
                feature_extraction_node.sonar_sub.callback(msg)
            elif topic == GYRO_TOPIC:
                gyro_node.gyro_sub.callback(msg)

            # 使用 IMU 驱动时钟
            if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:

                clock_pub.publish(Clock(msg.header.stamp))

                # 发布 map 到 world 的变换，以便在 rviz 中以 z-down 坐标系可视化所有内容
                node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")
    
    print(f"10 秒的话题和消息内容已成功写入 {output_txt_path}")
    

if __name__ == "__main__":

    # 初始化节点
    rospy.init_node("slam", log_level=rospy.INFO)

    # 调用类构造函数
    node = SLAMNode()
    node.init_node()

    # 解析参数并启动
    args, _ = common_parser().parse_known_args()

    if not args.file:
        loginfo("开始在线 SLAM...")
        rospy.spin()
    else:
        loginfo("开始离线 SLAM...")
        offline(args)
