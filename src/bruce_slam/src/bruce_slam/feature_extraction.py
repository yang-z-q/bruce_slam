#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, Image
import cv_bridge
import ros_numpy

from bruce_slam.utils.io import *
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import apply_custom_colormap
#from bruce_slam.feature import FeatureExtraction
from bruce_slam import pcl
import matplotlib.pyplot as plt
from sonar_oculus.msg import OculusPing, OculusPingUncompressed
from scipy.interpolate import interp1d

from .utils import *
from .sonar import *

from bruce_slam.CFAR import CFAR

#from bruce_slam.bruce_slam import sonar

class FeatureExtraction(object):
    """特征提取类，用于处理声纳数据并提取特征点"""

    def __init__(self):
        """初始化特征提取器"""
        # 声纳信息
        self.oculus = OculusProperty()

        # 初始化参数
        self.rows = None
        self.cols = None
        self.width = None
        self.height = None
        self.map_x = None
        self.map_y = None
        self.detector = None
        self.alg = None
        self.threshold = None
        self.resolution = None
        self.outlier_filter_min_points = None
        self.min_density = None
        self.skip = None
        self.compressed_images = None
        self.feature_img = None

        # 点云默认参数
        self.colormap = "RdBu_r"
        self.pub_rect = True
        self.res = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1
        self.maxRange = None

        # 多机器人系统的占位符
        self.rov_id = ""

    def configure(self):
        """配置特征提取参数"""
        # 从 ROS 参数服务器读取配置
        self.rows = rospy.get_param("~rows", 512)
        self.cols = rospy.get_param("~cols", 512)
        self.width = rospy.get_param("~width", 20.0)
        self.height = rospy.get_param("~height", 20.0)
        self.alg = rospy.get_param("~alg", "CA")
        self.threshold = rospy.get_param("~threshold", 30)
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.outlier_filter_min_points = rospy.get_param("~outlier_filter_min_points", 5)
        self.min_density = rospy.get_param("~min_density", 0.1)
        self.skip = rospy.get_param("~skip", 1)
        self.compressed_images = rospy.get_param("~compressed_images", True)

        # CV 桥接
        self.BridgeInstance = cv_bridge.CvBridge()
        
        # 读取格式
        self.coordinates = rospy.get_param(
            "~visualization/coordinates", "cartesian"
        )

        # 可视化参数
        self.radius = rospy.get_param("~visualization/radius")
        self.color = rospy.get_param("~visualization/color")

        # 声纳订阅者
        if self.compressed_images:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC, OculusPing, self.callback, queue_size=10)
        else:
            self.sonar_sub = rospy.Subscriber(
                SONAR_TOPIC_UNCOMPRESSED, OculusPingUncompressed, self.callback, queue_size=10)

        # 特征发布话题
        self.feature_pub = rospy.Publisher(
            SONAR_FEATURE_TOPIC, PointCloud2, queue_size=10)

        # 可视化发布话题
        self.feature_img_pub = rospy.Publisher(
            SONAR_FEATURE_IMG_TOPIC, Image, queue_size=10)

        self.detector = CFAR(self.rows, self.cols)

    def generate_map_xy(self, ping):
        """生成从极坐标到笛卡尔坐标的映射网格"""
        # 计算角度和距离
        bearings = np.linspace(
            ping.bearings[0], ping.bearings[-1], self.cols)
        ranges = np.linspace(ping.ranges[0], ping.ranges[-1], self.rows)

        # 创建网格
        bearing_grid, range_grid = np.meshgrid(bearings, ranges)

        # 转换为笛卡尔坐标
        x = range_grid * np.sin(bearing_grid)
        y = range_grid * np.cos(bearing_grid)

        # 归一化到图像坐标
        self.map_x = ((x + self.width/2) / self.width) * self.cols
        self.map_y = (y / self.height) * self.rows

    def publish_features(self, ping, points):
        """发布特征点云"""
        # 创建点云消息
        feature_msg = PointCloud2()
        # 设置时间戳和坐标系
        feature_msg.header.stamp = ping.header.stamp
        feature_msg.header.frame_id = "base_link"
        # 发布点云
        self.feature_pub.publish(feature_msg)

    def callback(self, sonar_msg):
        """特征提取回调函数
        参数:
            sonar_msg: OculusPing 消息，极坐标格式
        """
        if sonar_msg.ping_id % self.skip != 0:
            self.feature_img = None
            # 不是每一帧都提取特征
            # 但仍需要空点云用于 SLAM 节点的同步
            nan = np.array([[np.nan, np.nan]])
            self.publish_features(sonar_msg, nan)
            return

        # 解码压缩图像
        if self.compressed_images == True:
            img = np.frombuffer(sonar_msg.ping.data,np.uint8)
            img = np.array(cv2.imdecode(img,cv2.IMREAD_COLOR)).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # 图像未压缩，直接使用 ros_numpy
        else:
            img = ros_numpy.image.image_to_numpy(sonar_msg.ping)

        # 生成极坐标到笛卡尔坐标的映射网格
        self.generate_map_xy(sonar_msg)

        # 使用 CFAR 检测目标并检查阈值（极坐标）
        peaks = self.detector.detect(img, self.alg)
        peaks &= img > self.threshold

        # 可视化图像
        vis_img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        vis_img = cv2.applyColorMap(vis_img, 2)
        self.feature_img_pub.publish(ros_numpy.image.numpy_to_image(vis_img, "bgr8"))

        # 转换为笛卡尔坐标
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)        
        locs = np.c_[np.nonzero(peaks)]

        # 从图像坐标转换为米
        x = locs[:,1] - self.cols / 2.
        x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.)))
        y = (-1*(locs[:,0] / float(self.rows)) * self.height) + self.height
        points = np.column_stack((y,x))

        # 使用 PCL 降采样点云
        if len(points) and self.resolution > 0:
            points = pcl.downsample(points, self.resolution)

        # 移除离群点
        if self.outlier_filter_min_points > 1 and len(points) > 0:
            points = pcl.remove_outlier(points, self.outlier_filter_min_points, self.min_density)

        # 发布特征点
        self.publish_features(sonar_msg, points)
