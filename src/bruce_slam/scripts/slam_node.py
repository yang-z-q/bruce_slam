#!/usr/bin/env python

import rospy
from bruce_slam.utils.io import *
from bruce_slam.slam_ros import SLAMNode
from bruce_slam.utils.topics import *
from bruce_slam.utils.io import callback_lock_event
import os

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
    # gyro_node = GyroFilter()
    # gyro_node.init_node(SLAM_NS + "gyro/")
    """mp_node = MappingNode()
    mp_node.init_node(SLAM_NS + "mapping/")"""
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)

    # 创建输出文件
    # output_txt_path = "/home/yzq/sonar-SLAM/src/bruce_slam/scripts/bag_messages.txt"
    # start_time = None
    # txt_file = None
    # data_collected = False  # 添加标志来跟踪是否已收集数据
    # try:
    #     txt_file = open(output_txt_path, 'w', encoding='utf-8')
    #     # 遍历整个 rosbag
    #     for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
    #         while not rospy.is_shutdown():
    #             if callback_lock_event.wait(1.0):
    #                 break

    #         if rospy.is_shutdown():
    #             break

    #         # 记录起始时间
    #         if start_time is None:
    #             start_time = msg.header.stamp.to_sec()

    #         # 计算当前时间与起始时间的差值
    #         elapsed_time = msg.header.stamp.to_sec() - start_time
    #         if elapsed_time <= 3 and not data_collected and topic != SONAR_TOPIC_UNCOMPRESSED:  # 只保存前1秒的数据，且只保存一次
    #             # 将话题和消息内容写入 TXT 文件
    #             txt_file.write(f"Topic: {topic}\n")
    #             txt_file.write(f"Time: {msg.header.stamp.to_sec()}\n")
    #             txt_file.write(f"Message: {msg}\n")
    #             txt_file.write("-" * 80 + "\n")
    #             # 立即刷新缓冲区，确保数据写入文件
    #             txt_file.flush()
    #         elif elapsed_time > 1 and not data_collected:
    #             # 如果超过1秒，标记数据已收集，但继续运行
    #             print(f"已收集1秒数据,文件已保存到 {output_txt_path}")
    #             data_collected = True
    #             txt_file.close()  # 关闭文件
    #             txt_file = None   # 清空文件句柄

    #         if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
    #             dead_reckoning_node.imu_sub.callback(msg)
    #         elif topic == DVL_TOPIC:
    #             dead_reckoning_node.dvl_sub.callback(msg)
    #         elif topic == DEPTH_TOPIC:
    #             dead_reckoning_node.depth_sub.callback(msg)
    #         elif topic == SONAR_TOPIC or SONAR_TOPIC_UNCOMPRESSED:
    #             feature_extraction_node.sonar_sub.callback(msg)
    #         elif topic == GYRO_TOPIC:
    #             gyro_node.gyro_sub.callback(msg)

    #         # 使用 IMU 驱动时钟
    #         if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
    #             clock_pub.publish(Clock(msg.header.stamp))
    #             # 发布 map 到 world 的变换，以便在 rviz 中以 z-down 坐标系可视化所有内容
    #             node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")

    # except Exception as e:
    #     print(f"写入文件时发生错误: {str(e)}")
    # finally:
    #     if txt_file is not None:
    #         txt_file.close()
    #         print(f"前1秒的话题和消息内容已成功写入 {output_txt_path}")
    for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
        while not rospy.is_shutdown():
            if callback_lock_event.wait(1.0):
                break

        if rospy.is_shutdown():
            break

        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:
            dead_reckoning_node.imu_sub.callback(msg)
        elif topic == DVL_TOPIC:
            write_msg_to_file(msg, "dvl")
            dead_reckoning_node.dvl_sub.callback(msg)
        elif topic == DEPTH_TOPIC:
            write_msg_to_file(msg, "depth")
            dead_reckoning_node.depth_sub.callback(msg)
        elif topic == SONAR_TOPIC or SONAR_TOPIC_UNCOMPRESSED:
            feature_extraction_node.sonar_sub.callback(msg)
        # elif topic == GYRO_TOPIC:
        #     gyro_node.gyro_sub.callback(msg)

        # use the IMU to drive the clock
        if topic == IMU_TOPIC or topic == IMU_TOPIC_MK_II:

            clock_pub.publish(Clock(msg.header.stamp))

            # Publish map to world so we can visualize all in a z-down frame in rviz.
            node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")

def write_msg_to_file(msg, topic_name, filename="/home/yzq/sonar-SLAM/src/bruce_slam/scripts"):
    """
    将任意话题消息写入文件，每次运行前清空文件内容
    
    Args:
        msg: 消息内容
        topic_name: 话题名称，用于生成文件名
        filename: 输出文件目录
    """
    try:
        # 根据话题名称生成文件名
        output_file = os.path.join(filename, f"{topic_name.replace('/', '_')}_data.txt")
        
        # 第一次调用时清空文件
        if not hasattr(write_msg_to_file, 'initialized_files'):
            write_msg_to_file.initialized_files = set()
            
        if output_file not in write_msg_to_file.initialized_files:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("")  # 清空文件
            write_msg_to_file.initialized_files.add(output_file)
            
        # 使用追加模式写入
        with open(output_file, "a", encoding="utf-8") as f:
            # 写入完整消息内容
            f.write(f"{msg}\n")
            f.write("-" * 80 + "\n")
            
    except Exception as e:
        logerror(f"Error writing {topic_name} data: {str(e)}")

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
