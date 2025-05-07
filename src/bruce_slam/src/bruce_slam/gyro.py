import numpy as np
import rospy
from nav_msgs.msg import Odometry
import gtsam
from scipy.spatial.transform import Rotation

from kvh_gyro.msg import gyro

from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *

from std_msgs.msg import String, Float32


class GyroFilter(object):
	'''使用 DVL 和 IMU 读数进行航位推算的类
	'''
	def __init__(self):
		# 初始化欧拉角
		self.roll, self.yaw, self.pitch = 90.,0.,0.

	def init_node(self, ns:str="~")->None:
		"""节点初始化，获取所有相关参数等。

		参数:
			ns (str, 可选): 节点所在的命名空间。默认为 "~"。
		"""

		# 定义陀螺仪的旋转偏移矩阵，使陀螺仪坐标系与声纳坐标系对齐
		x = rospy.get_param(ns + "offset/x")
		y = rospy.get_param(ns + "offset/y")
		z = rospy.get_param(ns + "offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()

		# 地球自转速度
		self.latitude = np.radians(rospy.get_param(ns + "latitude"))
		self.earth_rate = -15.04107 * np.sin(self.latitude) / 3600.0
		self.sensor_rate = rospy.get_param(ns + "sensor_rate")

		# 定义 tf 转换器和陀螺仪订阅者
		self.odom_pub = rospy.Publisher(GYRO_INTEGRATION_TOPIC, Odometry, queue_size=self.sensor_rate+50)
		self.gyro_sub = rospy.Subscriber(GYRO_TOPIC, gyro, self.callback, queue_size=self.sensor_rate+50)

		loginfo("陀螺仪滤波节点已初始化")


	def callback(self, gyro_msg:gyro)->None:
		"""回调函数，接收原始陀螺仪读数（角度增量）并
		更新欧拉角估计。将这些角度作为 ROS 里程计消息发布。

		参数:
			gyro_msg (gyro): 输入的陀螺仪消息，这些是角度增量而不是旋转速率。
		"""

		# 解析消息并应用偏移矩阵
		dx,dy,dz = list(gyro_msg.delta)
		arr = np.array([dx,dy,dz])
		arr = arr.dot(self.offset_matrix)
		delta_yaw, delta_pitch, delta_roll = arr

		# 减去地球自转的影响
		delta_roll += (self.earth_rate / self.sensor_rate)

		# 执行积分，注意这是弧度制
		self.pitch += delta_pitch
		self.yaw += delta_yaw
		self.roll += delta_roll

		# 打包为 gtsam 对象
		rot = gtsam.Rot3.Ypr(self.yaw,self.pitch,self.roll)
		pose = gtsam.Pose3(rot, gtsam.Point3(0,0,0))

		# 发布里程计消息
		header = rospy.Header()
		header.stamp = gyro_msg.header.stamp
		header.frame_id = "odom"
		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(pose)
		odom_msg.child_frame_id = "base_link"
		odom_msg.twist.twist.linear.x = 0
		odom_msg.twist.twist.linear.y = 0
		odom_msg.twist.twist.linear.z = 0
		odom_msg.twist.twist.angular.x = 0
		odom_msg.twist.twist.angular.y = 0
		odom_msg.twist.twist.angular.z = 0
		self.odom_pub.publish(odom_msg)
