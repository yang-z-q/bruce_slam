#python imports
import tf
import rospy
import gtsam
import numpy as np
import rospy
from scipy.spatial.transform import Rotation


# standard ros message imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

# import custom messages
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth
from kvh_gyro.msg import gyro

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *

class KalmanNode(object):
	'''使用 DVL、IMU、光纤陀螺仪和深度读数的卡尔曼滤波类。
	'''

	def __init__(self):

		# 状态向量 = (x,y,z,横滚角,俯仰角,偏航角,x速度,y速度,z速度,横滚角速度,俯仰角速度,偏航角速度)
		self.state_vector= np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
		self.cov_matrix= np.diag([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
		self.yaw_gyro = 0.
		self.imu_yaw0 = None


	def init_node(self, ns="~")->None:
		"""初始化节点，获取所有参数。

		参数:
			ns (str, 可选): 节点的命名空间。默认为 "~"。
		"""

		self.state_vector = rospy.get_param(ns + "state_vector")
		self.cov_matrix = rospy.get_param(ns + "cov_matrix")
		self.R_dvl = rospy.get_param(ns + "R_dvl")
		self.dt_dvl = rospy.get_param(ns + "dt_dvl")
		self.H_dvl = np.array(rospy.get_param(ns + "H_dvl"))
		self.R_imu = rospy.get_param(ns + "R_imu")
		self.dt_imu = rospy.get_param(ns + "dt_imu")
		self.H_imu = np.array(rospy.get_param(ns + "H_imu"))
		self.H_gyro = np.array(rospy.get_param(ns + "H_gyro"))
		self.R_gyro = rospy.get_param(ns + "R_gyro")
		self.dt_gyro = rospy.get_param(ns + "dt_gyro")
		self.H_depth = np.array(rospy.get_param(ns + "H_depth"))
		self.R_depth = rospy.get_param(ns + "R_depth")
		self.dt_depth = rospy.get_param(ns + "dt_depth")
		self.Q = rospy.get_param(ns + "Q") # 过程噪声不确定性
		self.A_imu = rospy.get_param(ns + "A_imu") # 状态转移矩阵
		x = rospy.get_param(ns + "offset/x") # 陀螺仪偏移矩阵
		y = rospy.get_param(ns + "offset/y")
		z = rospy.get_param(ns + "offset/z")
		self.offset_matrix = Rotation.from_euler("xyz",[x,y,z],degrees=True).as_matrix()
		self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
		self.use_gyro = rospy.get_param(ns + "use_gyro")
		self.imu_offset = np.radians(rospy.get_param(ns + "imu_offset"))

		# 检查我们使用的是哪个版本的 IMU
		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu,callback=self.imu_callback,queue_size=250)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = rospy.Subscriber(IMU_TOPIC_MK_II, Imu, callback=self.imu_callback,queue_size=250)

		# 定义其他订阅者
		self.dvl_sub = rospy.Subscriber(DVL_TOPIC,DVL,callback=self.dvl_callback,queue_size=250)
		self.depth_sub = rospy.Subscriber(DEPTH_TOPIC, Depth,callback=self.pressure_callback,queue_size=250)
		self.odom_pub_kalman = rospy.Publisher(LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=250)

		# 定义变换广播器
		self.tf1 = tf.TransformBroadcaster()

		# 如果使用陀螺仪，设置订阅者
		if self.use_gyro:
			self.gyro_sub = rospy.Subscriber(GYRO_TOPIC, gyro, self.gyro_callback, queue_size=250)

		# 定义初始位姿，全部为零
		R_init = gtsam.Rot3.Ypr(0.,0.,0.)
		self.pose = gtsam.Pose3(R_init, gtsam.Point3(0, 0, 0))

		# 在 ROS 级别记录初始化完成
		loginfo("卡尔曼节点已初始化")


	def kalman_predict(self,previous_x:np.array,previous_P:np.array,A:np.array):
		"""传播状态和误差协方差。

		参数:
			previous_x (np.array): 前一个状态向量的值
			previous_P (np.array): 前一个协方差矩阵的值
			A (np.array): 状态转移矩阵

		返回:
			predicted_x (np.array): 预测估计
			predicted_P (np.array): 预测协方差矩阵
		"""

		A = np.array(A)
		predicted_P = A @ previous_P @ A.T + self.Q
		predicted_x = A @ previous_x

		return predicted_x, predicted_P


	def kalman_correct(self, predicted_x:np.array, predicted_P:np.array, z:np.array, H:np.array, R:np.array):
		"""测量更新。

		参数:
			predicted_x (np.array): 使用 kalman_predict() 预测的状态向量
			predicted_P (np.array): 使用 kalman_predict() 预测的协方差矩阵
			z (np.array): 输出向量（测量值）
			H (np.array): 观测矩阵（H_dvl, H_imu, H_gyro, H_depth）
			R (np.array): 测量不确定性（R_dvl, R_imu, R_gyro, R_depth）

		返回:
			corrected_x (np.array): 修正后的估计
			corrected_P (np.array): 修正后的协方差矩阵

		"""

		K = predicted_P @ H.T @ np.linalg.inv(H @ predicted_P @ H.T + R)
		corrected_x = predicted_x + K @ (z - H @ predicted_x)
		corrected_P = predicted_P - K @ H @ predicted_P

		return corrected_x, corrected_P


	def gyro_callback(self,gyro_msg:gyro)->None:
		"""仅使用光纤陀螺仪处理卡尔曼滤波。
		参数:
			gyro_msg (gyro): 来自陀螺仪的欧拉角
		"""

		# 解析消息并应用偏移矩阵
		arr = np.array(list(gyro_msg.delta))
		arr = arr.dot(self.offset_matrix) 
		delta_yaw_meas = np.array([[arr[0]],[0],[0]]) # 形状为(3,1)的测量值，用于应用卡尔曼滤波
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, delta_yaw_meas, self.H_gyro, self.R_gyro)
		self.yaw_gyro += self.state_vector[11][0]

	def dvl_callback(self, dvl_msg:DVL)->None:
		"""仅使用 DVL 处理卡尔曼滤波。

		参数:
			dvl_msg (DVL): 来自 DVL 的消息
		"""

		# 解析 DVL 速度
		dvl_measurement = np.array([[dvl_msg.velocity.x], [dvl_msg.velocity.y], [dvl_msg.velocity.z]])

		# 如果速度过高，不进行卡尔曼修正
		if np.any(np.abs(dvl_measurement) > self.dvl_max_velocity):
			return
		else:
			self.state_vector,self.cov_matrix  = self.kalman_correct(self.state_vector, self.cov_matrix, dvl_measurement, self.H_dvl, self.R_dvl)


	def pressure_callback(self,depth_msg:Depth):
		"""使用深度处理卡尔曼滤波。
		参数:
			depth_msg (Depth): 压力
		"""

		depth = np.array([[depth_msg.depth],[0],[0]]) # 我们需要形状为(3,1)用于修正
		self.state_vector,self.cov_matrix = self.kalman_correct(self.state_vector, self.cov_matrix, depth, self.H_depth, self.R_depth)

	def imu_callback(self, imu_msg:Imu)->None:
		"""仅使用 VN100 处理卡尔曼滤波。发布状态向量。

		参数:
			imu_msg (Imu): 来自 VN100 的消息
		"""

		# 卡尔曼预测
		predicted_x, predicted_P = self.kalman_predict(self.state_vector, self.cov_matrix, self.A_imu)

		# 解析 IMU 测量值
		roll_x, pitch_y, yaw_z = euler_from_quaternion((imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w))
		euler_angle = np.array([[self.imu_offset+roll_x], [pitch_y], [yaw_z]])

		# 如果还没有偏航角，将这个设为零点
		if self.imu_yaw0 is None:
			self.imu_yaw0 = yaw_z
		
		# 使偏航角相对于第一次测量
		euler_angle[2] -= self.imu_yaw0

		# 卡尔曼修正
		self.state_vector,self.cov_matrix = self.kalman_correct(predicted_x, predicted_P, euler_angle, self.H_imu, self.R_imu)

		# 使用滤波后的速度更新我们的 x 和 y 估计
		trans_x = self.state_vector[6][0]*self.dt_imu # x 更新
		trans_y = self.state_vector[7][0]*self.dt_imu # y 更新
		local_point = gtsam.Point2(trans_x, trans_y)

		# 检查是否使用光纤陀螺仪
		if self.use_gyro:
			R = gtsam.Rot3.Ypr(self.yaw_gyro,self.state_vector[4][0], self.state_vector[3][0]) 
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.yaw_gyro)
		else: # 不使用陀螺仪
			R = gtsam.Rot3.Ypr(self.state_vector[5][0], self.state_vector[4][0], self.state_vector[3][0])
			pose2 = gtsam.Pose2(self.pose.x(), self.pose.y(), self.pose.rotation().yaw())

		# 更新我们的位姿估计并发送里程计消息
		point = pose2.transformFrom(local_point)
		self.pose = gtsam.Pose3(R, gtsam.Point3(point[0], point[1], 0))
		self.send_odometry(imu_msg.header.stamp)

	def send_odometry(self,t:float):
		"""发布位姿。
		参数:
			t (float): 来自 imu_msg 的时间
		"""
		
		header = rospy.Header()
		header.stamp = t
		header.frame_id = "odom"
		odom_msg = Odometry()
		odom_msg.header = header
		odom_msg.pose.pose = g2r(self.pose)
		odom_msg.child_frame_id = "base_link"
		odom_msg.twist.twist.linear.x = 0.
		odom_msg.twist.twist.linear.y = 0.
		odom_msg.twist.twist.linear.z = 0.
		odom_msg.twist.twist.angular.x = 0.
		odom_msg.twist.twist.angular.y = 0.
		odom_msg.twist.twist.angular.z = 0.
		self.odom_pub_kalman.publish(odom_msg)

		self.tf1.sendTransform(
			(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z),
			(odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w),
			header.stamp, "base_link", "odom")
