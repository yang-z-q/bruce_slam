from ctypes import Union
from typing import Any
from enum import Enum

import gtsam
import numpy as np

from .sonar import OculusProperty
from .utils.conversions import *
from .utils.visualization import *
from .utils.io import *


class STATUS(Enum):
    """ICP 调用状态类"""

    NOT_ENOUGH_POINTS = "点数不足"
    LARGE_TRANSFORMATION = "变换过大"
    NOT_ENOUGH_OVERLAP = "重叠区域不足"
    NOT_CONVERGED = "未收敛"
    INITIALIZATION_FAILURE = "初始化失败"
    SUCCESS = "成功"

    def __init__(self, *args, **kwargs):
        """类构造函数"""
        Enum.__init__(*args, **kwargs)
        self.description = None

    def __bool__(self) -> bool:
        """布尔值重载

        返回:
            bool: 如果状态为 SUCCESS 则返回 true,否则返回 false
        """
        return self == STATUS.SUCCESS

    def __nonzero__(self) -> bool:
        """布尔值重载

        返回:
            bool: 如果状态为 SUCCESS 则返回 true,否则返回 false
        """
        return self == STATUS.SUCCESS

    def __str__(self) -> str:
        """将类转换为字符串

        返回:
            str: 类的可打印版本
        """
        if self.description:
            return self.value + ": " + self.description
        else:
            return self.value


class Keyframe(object):
    """SLAM 的关键帧对象。在这个 SLAM 解决方案中，我们使用因子图和关键帧列表。
    关键帧存储从更新后的位姿到该时间步观察到的所有点。
    """

    def __init__(
        self,
        status: bool,
        time: rospy.Time,
        dr_pose3: gtsam.Pose3,
        points: np.array = np.zeros((0, 2), np.float32),
        cov: np.array = None,
        source_pose=None, 
        between_pose=None, 
        index=None, 
        vin=None, 
        index_kf=None
    ):
        """关键帧类的构造函数

        参数:
            status (bool): 这个帧是否是关键帧？
            time (rospy.Time): 输入消息的时间戳
            dr_pose3 (gtsam.Pose3): 航迹推算位姿
            points (np.array, 可选): 点云数组。默认为 np.zeros((0, 2), np.float32)。
            cov (np.array, 可选): 协方差矩阵。默认为 None。
        """

        self.status = status  # 用于标记关键帧
        self.time = time  # 时间

        self.dr_pose3 = dr_pose3  # 航迹推算 3D 位姿
        self.dr_pose = pose322(dr_pose3)  # 航迹推算 2D 位姿

        self.pose3 = dr_pose3  # 估计的 3D 位姿（稍后更新）
        self.pose = pose322(dr_pose3)  # 估计的 2D 位姿

        self.cov = cov  # 局部坐标系中的协方差（始终为 2D）
        self.transf_cov = None  # 全局坐标系中的协方差

        self.points = points.astype(np.float32)  # 局部坐标系中的点（始终为 2D）
        self.transf_points = None  # 基于位姿转换到全局坐标系的点

        self.points3D = points.astype(
            np.float32
        )  # 来自正交传感器融合的 3D 点云
        self.transf_points3D = (
            None  # 基于位姿转换到全局坐标系的 3D 点云
        )

        self.constraints = (
            []
        )  # 非顺序约束（键，里程计）即回环检测

        self.twist = None  # 发布里程计的 twist 消息

        self.image = None  # 此关键帧的图像
        self.vertical_images = []
        self.horizontal_images = []

        self.poseTrue = None  # 记录 gazebo 中的真实位姿，仅用于仿真

        self.sub_frames = []

        self.submap = None # 多机器人 SLAM 数据
        self.ring_key = None
        self.context = None
        self.redo_submap = False
        self.source_pose = source_pose
        self.between_pose = between_pose
        self.index = index
        self.guess_pose = None
        self.vin = vin
        self.index_kf = index_kf
        self.scan_match_prediction = None
        self.scan_match_prediction_status = False
        self.scan_match_eig_max = None
        self.bits = None

    def update(self, new_pose: gtsam.Pose2, new_cov: np.array = None) -> None:
        """在 SLAM 更新后更新关键帧，传入新的位姿和协方差

        参数:
            new_pose (gtsam.Pose2): SLAM 优化后的新位姿
            new_cov (np.array, 可选): 新的协方差矩阵。默认为 None。
        """

        # 更新 2D 和 3D 位姿
        self.pose = new_pose
        self.pose3 = n2g(
            (
                new_pose.x(),
                new_pose.y(),
                self.dr_pose3.z(),
                self.dr_pose3.rotation().roll(),
                self.dr_pose3.rotation().pitch(),
                new_pose.theta(),
            ),
            "Pose3",
        )

        # 基于新位姿转换点云，2D 和 3D
        self.transf_points = Keyframe.transform_points(self.points, self.pose)
        self.transf_points3D = Keyframe.transform_points_3D(
            self.points3D, self.pose, self.pose3
        )

        # 如果有新的协方差则更新
        if new_cov is not None:
            self.cov = new_cov

        # 将协方差转换到全局坐标系
        if self.cov is not None:
            c, s = np.cos(self.pose.theta()), np.sin(self.pose.theta())
            R = np.array([[c, -s], [s, c]])
            self.transf_cov = np.array(self.cov)
            self.transf_cov[:2, :2] = R.dot(self.transf_cov[:2, :2]).dot(R.T)
            self.transf_cov[:2, 2] = R.dot(self.transf_cov[:2, 2])
            self.transf_cov[2, :2] = self.transf_cov[2, :2].dot(R.T)

    @staticmethod
    def transform_points(points: np.array, pose: gtsam.Pose2) -> np.array:
        """给定位姿转换一组 2D 点

        参数:
            points (np.array): 要转换的点云
            pose (gtsam.Pose2): 要应用的变换

        返回:
            np.array: 转换后的点云
        """

        # 检查是否有点
        if len(points) == 0:
            return np.empty_like(points, np.float32)

        # 将位姿转换为矩阵格式
        T = pose.matrix().astype(np.float32)

        # 旋转和平移到全局坐标系
        return points.dot(T[:2, :2].T) + T[:2, 2]

    @staticmethod
    def transform_points_3D(
        points: np.array, pose: gtsam.Pose2, pose3: gtsam.Pose3
    ) -> np.array:
        """将一组 3D 点转换到给定位姿

        参数:
            points (np.array): 要转换的点
            pose (gtsam.Pose2): 要移除的 2D 位姿
            pose3 (gtsam.Pose3): 要应用的 3D 变换

        返回:
            np.array: 转换后的点云
        """

        # 检查是否有点
        if len(points) == 0 or points.shape[1] != 3:
            return np.empty_like(points, np.float32)

        # 将位姿转换为矩阵格式
        H = pose3.matrix().astype(np.float32)

        # 旋转和平移到全局坐标系
        return points.dot(H[:3, :3].T) + H[:3, 3]


class InitializationResult(object):
    """存储全局 ICP 所需的所有内容"""

    def __init__(self):
        """类构造函数"""

        # 所有点都在局部坐标系中
        self.source_points = np.zeros((0, 2))
        self.target_points = np.zeros((0, 2))
        self.source_key = None
        self.target_key = None
        self.source_pose = None
        self.target_pose = None
        # 用于采样的协方差
        self.cov = None
        self.occ = None
        self.status = None
        self.estimated_source_pose = None
        self.source_pose_samples = None


class ICPResult(object):
    """存储 ICP 的结果"""

    def __init__(
        self,
        init_ret: InitializationResult,
        use_samples: bool = False,
        sample_eps: float = 0.01,
    ):
        """类构造函数

        参数:
            init_ret (InitializationResult): 初始化结果
            use_samples (bool, 可选): 是否使用采样。默认为 False。
            sample_eps (float, 可选): 采样精度。默认为 0.01。
        """

        # 所有点都在局部坐标系中
        self.source_points = init_ret.source_points
        self.target_points = init_ret.target_points
        self.source_key = init_ret.source_key
        self.target_key = init_ret.target_key
        self.source_pose = init_ret.source_pose
        self.target_pose = init_ret.target_pose
        self.status = init_ret.status
        self.estimated_transform = None
        self.cov = None
        self.initial_transforms = None
        self.inserted = False
        self.sample_transforms = None

        # 填充初始变换
        if init_ret.estimated_source_pose is not None:
            self.initial_transform = self.target_pose.between(
                init_ret.estimated_source_pose
            )
        else:
            self.initial_transform = self.target_pose.between(self.source_pose)

        # 如果使用采样来推导协方差矩阵
        if use_samples and init_ret.source_pose_samples is not None:
            idx = np.argsort(init_ret.source_pose_samples[:, -1])
            transforms = [
                self.target_pose.between(n2g(g, "Pose2"))
                for g in init_ret.source_pose_samples[idx, :3]
            ]
            filtered = [transforms[0]]
            for b in transforms[1:]:
                d = np.linalg.norm(g2n(filtered[-1].between(b)))
                if d < sample_eps:
                    continue
                else:
                    filtered.append(b)
            self.initial_transforms = filtered


class SMParams(object):
    """扫描匹配参数类"""

    def __init__(self):
        """构造函数"""

        # 使用占据概率地图匹配来初始化 ICP
        self.initialization = None
        # 全局搜索参数
        self.initialization_params = None
        # 最小点数
        self.min_points = None
        # 与初始猜测的最大偏差
        self.max_translation = None
        self.max_rotation = None

        # 源关键帧与最后一个目标帧之间的最小间隔
        self.min_st_sep = None
        # 用于构建源点的源帧数量
        # 在 SSM 中不使用
        self.source_frames = None
        # 用于构建目标点的目标帧数量
        # 在 NSSM 中不使用
        self.target_frames = None

        # 用于计算协方差的 ICP 实例数量
        self.cov_samples = None
