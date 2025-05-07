# python imports
import threading
import tf
import rospy
import cv_bridge
from nav_msgs.msg import Odometry
from message_filters import  Subscriber
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer

# bruce imports
from bruce_slam.utils.io import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import *
from bruce_slam.slam import SLAM, Keyframe
from bruce_slam import pcl

# Argonaut imports
from sonar_oculus.msg import OculusPing


class SLAMNode(SLAM):
    """SLAM ROS 节点类，继承自 SLAM 基类"""
    
    def __init__(self):
        """初始化 SLAM 节点"""
        super(SLAMNode, self).__init__()
        self.tf = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.rov_id = ""

    def init_node(self, ns="~")->None:
        """初始化 ROS 节点
        参数:
            ns: 命名空间
        """
        # 读取配置参数
        self.rov_id = rospy.get_param(ns + "rov_id", "")
        self.enable_slam = rospy.get_param(ns + "enable_slam", True)
        self.save_fig = rospy.get_param(ns + "save_fig", False)
        self.save_data = rospy.get_param(ns + "save_data", False)

        # 初始化 SLAM 基类
        super(SLAMNode, self).init_node(ns)

        # 创建时间同步器
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub], 10, 0.1
        )

        # 注册回调函数
        self.time_sync.registerCallback(self.SLAM_callback)

        # 位姿发布器
        self.pose_pub = rospy.Publisher(
            SLAM_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=10)

        # 里程计发布器
        self.odom_pub = rospy.Publisher(SLAM_ODOM_TOPIC, Odometry, queue_size=10)

        # SLAM 轨迹发布器
        self.traj_pub = rospy.Publisher(
            SLAM_TRAJ_TOPIC, PointCloud2, queue_size=1, latch=True)

        # 位姿约束发布器
        self.constraint_pub = rospy.Publisher(
            SLAM_CONSTRAINT_TOPIC, Marker, queue_size=1, latch=True)

        # 点云发布器
        self.cloud_pub = rospy.Publisher(
            SLAM_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True)

    @add_lock
    def sonar_callback(self, ping:OculusPing)->None:
        """声纳数据回调函数
        参数:
            ping: 声纳数据消息
        """
        super(SLAMNode, self).sonar_callback(ping)

    @add_lock
    def SLAM_callback(self, feature_msg:PointCloud2, odom_msg:Odometry)->None:
        """SLAM 回调函数
        参数:
            feature_msg: 特征点云消息
            odom_msg: 里程计消息
        """
        super(SLAMNode, self).SLAM_callback(feature_msg, odom_msg)
        self.publish_all()

    def publish_all(self)->None:
        """发布所有 SLAM 结果"""
        self.publish_pose()
        self.publish_constraint()
        self.publish_trajectory()
        self.publish_point_cloud()

    def publish_pose(self)->None:
        """发布当前位姿"""
        if len(self.keyframes) == 0:
            return

        # 创建位姿消息
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            pose_msg.header.frame_id = "map"
        else:
            pose_msg.header.frame_id = self.rov_id + "_map"

        # 设置位姿
        pose = self.current_keyframe.pose
        pose_msg.pose.pose.position.x = pose[0]
        pose_msg.pose.pose.position.y = pose[1]
        pose_msg.pose.pose.position.z = 0
        pose_msg.pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, pose[2]))

        # 设置协方差
        pose_msg.pose.covariance = [0] * 36
        pose_msg.pose.covariance[0] = self.current_keyframe.covariance[0, 0]
        pose_msg.pose.covariance[7] = self.current_keyframe.covariance[1, 1]
        pose_msg.pose.covariance[35] = self.current_keyframe.covariance[2, 2]

        # 发布位姿
        self.pose_pub.publish(pose_msg)

        # 创建里程计消息
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose = pose_msg.pose

        # 发布里程计
        self.odom_pub.publish(odom_msg)

    def publish_constraint(self)->None:
        """发布位姿约束"""
        if len(self.keyframes) < 2:
            return

        # 创建约束标记
        link_msg = Marker()
        link_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            link_msg.header.frame_id = "map"
        else:
            link_msg.header.frame_id = self.rov_id + "_map"

        link_msg.ns = "constraints"
        link_msg.id = 0
        link_msg.type = Marker.LINE_LIST
        link_msg.action = Marker.ADD
        link_msg.scale.x = 0.1
        link_msg.color.r = 1.0
        link_msg.color.a = 1.0

        # 添加约束线
        for i in range(len(self.keyframes) - 1):
            p1 = self.keyframes[i].pose
            p2 = self.keyframes[i + 1].pose
            link_msg.points.append(Point(p1[0], p1[1], 0))
            link_msg.points.append(Point(p2[0], p2[1], 0))

        # 发布约束
        self.constraint_pub.publish(link_msg)

    def publish_trajectory(self)->None:
        """发布轨迹"""
        if len(self.keyframes) == 0:
            return

        # 创建轨迹点云
        traj = np.zeros((len(self.keyframes), 3))
        for i in range(len(self.keyframes)):
            traj[i, 0] = self.keyframes[i].pose[0]
            traj[i, 1] = self.keyframes[i].pose[1]
            traj[i, 2] = 0

        # 转换为点云消息
        traj_msg = n2r(traj, "PointCloudXYZ")
        traj_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            traj_msg.header.frame_id = "map"
        else:
            traj_msg.header.frame_id = self.rov_id + "_map"

        # 发布轨迹
        self.traj_pub.publish(traj_msg)

    def publish_point_cloud(self)->None:
        """发布降采样后的 3D 点云，z 坐标为 0
        最后一列表示观测到该点的关键帧索引
        """
        # 定义空数组
        all_points = [np.zeros((0, 2), np.float32)]
        all_keys = []

        # 遍历所有关键帧，将点云注册到原点
        for key in range(len(self.keyframes)):
            # 解析位姿
            pose = self.keyframes[key].pose
            # 获取转换后的点云
            transf_points = self.keyframes[key].transf_points
            # 添加点云和关键帧索引
            all_points.append(transf_points)
            all_keys.append(key * np.ones((len(transf_points), 1)))

        # 合并点云和索引
        all_points = np.concatenate(all_points)
        all_keys = np.concatenate(all_keys)

        # 使用 PCL 降采样点云
        sampled_points, sampled_keys = pcl.downsample(
            all_points, all_keys, self.point_resolution
        )

        # 转换为 ROS XYZI 格式
        sampled_xyzi = np.c_[sampled_points, np.zeros_like(sampled_keys), sampled_keys]
        
        # 如果没有点，直接返回
        if len(sampled_xyzi) == 0:
            return

        # 转换为 ROS 消息并发布
        cloud_msg = n2r(sampled_xyzi, "PointCloudXYZI")
        cloud_msg.header.stamp = self.current_keyframe.time
        if self.rov_id == "":
            cloud_msg.header.frame_id = "map"
        else:
            cloud_msg.header.frame_id = self.rov_id + "_map"
        self.cloud_pub.publish(cloud_msg)
