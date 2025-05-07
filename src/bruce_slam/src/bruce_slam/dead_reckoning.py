# python imports
import tf
import rospy
import gtsam
import numpy as np

# ros-python imports
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
from message_filters import ApproximateTimeSynchronizer, Cache, Subscriber

# import custom messages
from kvh_gyro.msg import gyro as GyroMsg
from rti_dvl.msg import DVL
from bar30_depth.msg import Depth

# bruce imports
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.utils.visualization import ros_colorline_trajectory

import math
from std_msgs.msg import String, Float32
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class DeadReckoningNode(object):
	'''使用 DVL 和 IMU 读数进行航位推算的类
	'''
	def __init__(self):
		self.pose = None # 车辆位姿
		self.prev_time = None # 上一次读数时间
		self.prev_vel = None # 上一次读数速度
		self.keyframes = [] # 关键帧列表

		# 强制原点处的偏航角与 x 轴对齐
		self.imu_yaw0 = None
		self.imu_pose = [0, 0, 0, -np.pi / 2, 0, 0]
		self.imu_rot = None
		self.dvl_max_velocity = 0.3

		# 在以下情况下创建新的关键位姿：
		# - |ti - tj| > min_duration 且
		# - |xi - xj| > max_translation 或
		# - |ri - rj| > max_rotation
		self.keyframe_duration = None
		self.keyframe_translation = None
		self.keyframe_rotation = None
		self.dvl_error_timer = 0.0

		# 多机器人 SLAM 的占位符
		self.rov_id = ""


	def init_node(self, ns="~")->None:
		"""初始化节点，从 ROS 获取所有参数

		参数:
			ns (str, 可选): 节点的命名空间。默认为 "~"。
		"""
		# 节点参数
		self.imu_pose = rospy.get_param(ns + "imu_pose")
		self.imu_pose = n2g(self.imu_pose, "Pose3")
		self.imu_rot = self.imu_pose.rotation()
		self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
		self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
		self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
		self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

		# 订阅者和缓存
		self.dvl_sub = Subscriber(DVL_TOPIC, DVL)
		self.gyro_sub = Subscriber(GYRO_INTEGRATION_TOPIC, Odometry)
		self.depth_sub = Subscriber(DEPTH_TOPIC, Depth)
		self.depth_cache = Cache(self.depth_sub, 1)

		if rospy.get_param(ns + "imu_version") == 1:
			self.imu_sub = Subscriber(IMU_TOPIC, Imu)
		elif rospy.get_param(ns + "imu_version") == 2:
			self.imu_sub = Subscriber(IMU_TOPIC_MK_II, Imu)

		# 使用点云进行可视化
		self.traj_pub = rospy.Publisher(
			"traj_dead_reck", PointCloud2, queue_size=10)

		self.odom_pub = rospy.Publisher(
			LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=10)

		# 是否使用光纤陀螺仪？
		self.use_gyro = rospy.get_param(ns + "use_gyro")

		# 定义回调函数，使用陀螺仪还是 VN100？
		if self.use_gyro:
			self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub, self.gyro_sub], 300, .1)
			self.ts.registerCallback(self.callback_with_gyro)
		else:
			self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub], 200, .1)
			self.ts.registerCallback(self.callback)

		self.tf = tf.TransformBroadcaster()

		loginfo("定位节点已初始化")


	def callback(self, imu_msg:Imu, dvl_msg:DVL)->None:
		"""仅使用 VN100 和 DVL 处理航位推算。融合并发布里程计消息。

		参数:
			imu_msg (Imu): VN100 的消息
			dvl_msg (DVL): DVL 的消息
		"""
		# 获取上一次深度消息
		depth_msg = self.depth_cache.getLast()
		# 如果没有深度消息，则跳过这个时间步
		if depth_msg is None:
			return

		# 检查深度消息和 DVL 之间的延迟
		dd_delay = (depth_msg.header.stamp - dvl_msg.header.stamp).to_sec()
		#print(dd_delay)
		if abs(dd_delay) > 1.0:
			logdebug("深度消息缺失 {} 秒".format(dd_delay))

		# 将 IMU 消息从消息格式转换为 gtsam 旋转对象
		rot = r2g(imu_msg.orientation)
		rot = rot.compose(self.imu_rot.inverse())

		# 如果还没有偏航角，将这个设为零点
		if self.imu_yaw0 is None:
			self.imu_yaw0 = rot.yaw()

		# 获取旋转矩阵
		# 如果 Kalman 和 DeadReck 中的 use_gyro 值相同，使用这行
		rot = gtsam.Rot3.Ypr(rot.yaw()-self.imu_yaw0, rot.pitch(), np.radians(90)+rot.roll())
		# 如果 Kalman 中 use_gyro = True 且 DeadReck 中 use_gyro = False，使用这行：
		# rot = gtsam.Rot3.Ypr(rot.yaw()-self.imu_yaw0, rot.pitch(), np.radians(90)+rot.roll())

		# 将 DVL 消息解析为速度数组
		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		# 打包里程计消息并发布
		self.send_odometry(vel,rot,dvl_msg.header.stamp,depth_msg.depth)


	def callback_with_gyro(self, imu_msg:Imu, dvl_msg:DVL, gyro_msg:GyroMsg)->None:
		"""使用光纤陀螺仪处理航位推算状态估计。这里我们使用
		陀螺仪作为获取偏航角估计的手段，横滚角和俯仰角仍然来自 VN100。

		参数:
			imu_msg (Imu): vn100 imu 消息
			dvl_msg (DVL): DVL 消息
			gyro_msg (GyroMsg): 来自陀螺仪的欧拉角
		"""
		# 解码陀螺仪消息
		gyro_yaw = r2g(gyro_msg.pose.pose).rotation().yaw()

		# 获取上一次深度消息
		depth_msg = self.depth_cache.getLast()

		# 如果没有深度消息，则跳过这个时间步
		if depth_msg is None:
			return

		# 检查深度消息和 DVL 之间的延迟
		dd_delay = (depth_msg.header.stamp - dvl_msg.header.stamp).to_sec()
		#print(dd_delay)
		if abs(dd_delay) > 1.0:
			logdebug("深度消息缺失 {} 秒".format(dd_delay))

		# 将 IMU 消息从消息格式转换为 gtsam 旋转对象
		rot = r2g(imu_msg.orientation)
		rot = rot.compose(self.imu_rot.inverse())


		# 获取旋转矩阵
		rot = gtsam.Rot3.Ypr(gyro_yaw, rot.pitch(), rot.roll())

		# 将 DVL 消息解析为速度数组
		vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])

		# 打包里程计消息并发布
		self.send_odometry(vel,rot,dvl_msg.header.stamp,depth_msg.depth)


	def send_odometry(self,vel:np.array,rot:gtsam.Rot3,dvl_time:rospy.Time,depth:float)->None:
		"""打包给定所有 DVL、旋转矩阵和深度的里程计

		参数:
			vel (np.array): DVL 速度的一维 numpy 数组
			rot (gtsam.Rot3): 车辆的旋转矩阵
			dvl_time (rospy.Time): DVL 消息的时间戳
			depth (float): 车辆深度
		"""

		# 如果 DVL 消息有任何速度超过最大阈值，进行错误处理
		if np.any(np.abs(vel) > self.dvl_max_velocity):
			if self.pose:

				self.dvl_error_timer += (dvl_time - self.prev_time).to_sec()
				if self.dvl_error_timer > 5.0:
					logwarn(
						"DVL 速度 ({:.1f}, {:.1f}, {:.1f}) 超过最大速度 {:.1f} 持续 {:.1f} 秒。".format(
							vel[0],
							vel[1],
							vel[2],
							self.dvl_max_velocity,
							self.dvl_error_timer,
						)
					)
				vel = self.prev_vel
			else:
				return
		else:
			self.dvl_error_timer = 0.0

		if self.pose:
			# 使用 DVL 消息计算我们在机体坐标系中移动的距离
			dt = (dvl_time - self.prev_time).to_sec()
			dv = (vel + self.prev_vel) * 0.5
			trans = dv * dt

			# 获取仅包含横滚角和俯仰角的旋转矩阵
			rotation_flat = gtsam.Rot3.Ypr(0, rot.pitch(), rot.roll())

			# 将我们的运动转换到全局坐标系
			#trans[2] = -trans[2]
			#trans = trans.dot(rotation_flat.matrix())

			# 使用 GTSAM 工具传播我们的运动
			local_point = gtsam.Point2(trans[0], trans[1])

			pose2 = gtsam.Pose2(
				self.pose.x(), self.pose.y(), self.pose.rotation().yaw()
			)
			point = pose2.transformFrom(local_point)

			self.pose = gtsam.Pose3(
				rot, gtsam.Point3(point[0], point[1], depth)
			)

		else:
			# 初始化位姿
			self.pose = gtsam.Pose3(rot, gtsam.Point3(0, 0, depth))

		# 记录这个时间步的消息供下次使用
		self.prev_time = dvl_time
		self.prev_vel = vel

		new_keyframe = False
		if not self.keyframes:
			new_keyframe = True
		else:
			duration = self.prev_time.to_sec() - self.keyframes[-1][0]
			if duration > self.keyframe_duration:
				odom = self.keyframes[-1][1].between(self.pose)
				odom = g2n(odom)
				translation = np.linalg.norm(odom[:3])
				rotation = abs(odom[-1])

				if (
					translation > self.keyframe_translation
					or rotation > self.keyframe_rotation
				):
					new_keyframe = True

		if new_keyframe:
			self.keyframes.append((self.prev_time.to_sec(), self.pose))
		self.publish_pose(new_keyframe)


	def publish_pose(self, publish_traj:bool=False)->None:
		"""Publish the pose

		Args:
			publish_traj (bool, optional): Are we publishing the whole set of keyframes?. Defaults to False.

		"""
		if self.pose is None:
			return

		header = rospy.Header()
		header.stamp = self.prev_time
		header.frame_id = "odom"

		odom_msg = Odometry()
		odom_msg.header = header
		# pose in odom frame
		odom_msg.pose.pose = g2r(self.pose)
		# twist in local frame
		odom_msg.child_frame_id = "base_link"
		# Local planer behaves worse
		# odom_msg.twist.twist.linear.x = self.prev_vel[0]
		# odom_msg.twist.twist.linear.y = self.prev_vel[1]
		# odom_msg.twist.twist.linear.z = self.prev_vel[2]
		# odom_msg.twist.twist.angular.x = self.prev_omega[0]
		# odom_msg.twist.twist.angular.y = self.prev_omega[1]
		# odom_msg.twist.twist.angular.z = self.prev_omega[2]
		odom_msg.twist.twist.linear.x = 0
		odom_msg.twist.twist.linear.y = 0
		odom_msg.twist.twist.linear.z = 0
		odom_msg.twist.twist.angular.x = 0
		odom_msg.twist.twist.angular.y = 0
		odom_msg.twist.twist.angular.z = 0
		self.odom_pub.publish(odom_msg)

		p = odom_msg.pose.pose.position
		q = odom_msg.pose.pose.orientation
		self.tf.sendTransform(
			(p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, "base_link", "odom"
		)
		if publish_traj:
			traj = np.array([g2n(pose) for _, pose in self.keyframes])
			traj_msg = ros_colorline_trajectory(traj)
			traj_msg.header = header
			self.traj_pub.publish(traj_msg)
