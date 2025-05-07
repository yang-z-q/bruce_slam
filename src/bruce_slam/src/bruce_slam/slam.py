import gtsam
import numpy as np

from ctypes import Union
from typing import Any
from numpy import True_
from scipy.optimize import shgo
from itertools import combinations
from collections import defaultdict
from sklearn.covariance import MinCovDet
import time as time_pkg

from .sonar import OculusProperty
from .utils.conversions import *
from .utils.visualization import *
from .utils.io import *
from . import pcl

from bruce_slam.slam_objects import (
    STATUS,
    Keyframe,
    InitializationResult,
    ICPResult,
    SMParams,
)


class SLAM(object):
    """基于水下声纳的 SLAM 类"""

    def __init__(self):
        """SLAM 类的构造函数，注意我们不使用 Python 的常规方式传入参数，
        而是使用 ROS 参数系统获取参数。请注意，当调用 yaml 文件时，几乎所有参数都可以被覆盖。
        详见 config/slam.yaml。"""

        # 配置声纳信息
        self.oculus = OculusProperty()

        # 在以下情况下创建新的因子：
        # - |ti - tj| > min_duration 且
        # - |xi - xj| > max_translation 或
        # - |ri - rj| > max_rotation
        self.keyframe_duration = None
        self.keyframe_translation = None
        self.keyframe_rotation = None

        # 关键帧列表，关键帧是 SLAM 解决方案中的一个步骤
        self.keyframes = []

        # 当前（非关键）帧，具有实时位姿更新
        # TODO 从前一个关键帧传播协方差
        self.current_frame = None

        # 初始化 isam 图优化工具
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)

        # 定义图和初始猜测矩阵、值。使用这些将信息推送到 isam
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # 初始位置噪声模型 [x, y, theta]
        self.prior_sigmas = None  # 占位符

        # 无 ICP 时的噪声模型，仅航位推算
        # [x, y, theta]
        self.odom_sigmas = None  # 占位符

        # 用于 ICP 和发布的点云下采样参数
        self.point_resolution = 0.5

        # 重叠估计中的噪声半径
        self.point_noise = 0.5

        # 顺序扫描匹配（SSM）的参数
        self.ssm_params = SMParams()  # 用于保存所有参数的对象
        self.ssm_params.initialization = True  # 指示是否完成此步骤的标志
        self.ssm_params.initialization_params = 50, 1, 0.01
        self.ssm_params.min_st_sep = 1
        self.ssm_params.min_points = 50
        self.ssm_params.max_translation = 2.0
        self.ssm_params.max_rotation = np.pi / 6
        self.ssm_params.target_frames = 3
        # 不使用 ICP 协方差
        self.ssm_params.cov_samples = 0

        # 回环检测（NSSM）的参数
        self.nssm_params = SMParams()
        self.nssm_params.initialization = True
        self.nssm_params.initialization_params = 100, 5, 0.01
        self.nssm_params.min_st_sep = 10
        self.nssm_params.min_points = 100
        self.nssm_params.max_translation = 6.0
        self.nssm_params.max_rotation = np.pi / 2
        self.nssm_params.source_frames = 5
        self.nssm_params.cov_samples = 30

        # 定义 ICP
        self.icp = pcl.ICP()
        self.icp_ssm = pcl.ICP()

        # 成对一致性测量，用于回环检测异常值剔除
        self.nssm_queue = []  # 回环检测队列
        self.pcm_queue_size = 5  # 默认值
        self.min_pcm = 3  # 默认值

        # 在两种情况下使用固定噪声模型：
        # - 顺序扫描匹配
        # - 非顺序扫描匹配中 ICP 协方差太小
        # [x, y, theta]
        self.icp_odom_sigmas = None

        # 在在线模式下无法保存图像
        # TODO 移除此项
        self.save_fig = False
        self.save_data = False

    @property
    def current_keyframe(self) -> Keyframe:
        """获取当前关键帧

        返回:
            Keyframe: 系统中的当前关键帧（最新关键帧）
        """
        return self.keyframes[-1]

    @property
    def current_key(self) -> int:
        """获取存储关键帧列表的长度

        返回:
            int: self.keyframes 的长度
        """
        return len(self.keyframes)

    def configure(self) -> None:
        """配置 SLAM"""

        # 检查 nssm 协方差参数
        assert (
            self.nssm_params.cov_samples == 0
            or self.nssm_params.cov_samples
            < self.nssm_params.initialization_params[0]
            * self.nssm_params.initialization_params[1]
        )

        # 检查 ssm 协方差参数
        assert (
            self.ssm_params.cov_samples == 0
            or self.ssm_params.cov_samples
            < self.ssm_params.initialization_params[0]
            * self.ssm_params.initialization_params[1]
        )

        assert self.nssm_params.source_frames < self.nssm_params.min_st_sep

        # 创建噪声模型
        self.prior_model = self.create_noise_model(self.prior_sigmas)
        self.odom_model = self.create_noise_model(self.odom_sigmas)
        self.icp_odom_model = self.create_noise_model(self.icp_odom_sigmas)

    def get_states(self) -> np.array:
        """获取所有状态作为数组，表示为
            [time, pose2, dr_pose3, cov]
            - pose2: [x, y, yaw]
            - dr_pose3: [x, y, z, roll, pitch, yaw]
            - cov: 3 x 3

        返回:
            np.array: 状态数组
        """
        # 构建状态数组
        states = np.zeros(
            self.current_key,
            dtype=[
                ("time", np.float64),
                ("pose", np.float32, 3),
                ("dr_pose3", np.float32, 6),
                ("cov", np.float32, 9),
            ],
        )

        # 更新所有关键帧
        values = self.isam.calculateEstimate()
        for key in range(self.current_key):
            pose = values.atPose2(X(key))
            cov = self.isam.marginalCovariance(X(key))
            self.keyframes[key].update(pose, cov)

        # 提取状态
        t_zero = self.keyframes[0].time
        for key in range(self.current_key):
            keyframe = self.keyframes[key]
            states[key]["time"] = (keyframe.time - t_zero).to_sec()
            states[key]["pose"] = g2n(keyframe.pose)
            states[key]["dr_pose3"] = g2n(keyframe.dr_pose3)
            states[key]["cov"] = keyframe.transf_cov.ravel()
        return states

    @staticmethod
    def sample_pose(pose: gtsam.Pose2, covariance: np.array) -> gtsam.Pose2:
        """使用协方差矩阵定义的正态分布生成随机位姿。

        参数:
            pose (gtsam.Pose2): 要添加随机噪声的位姿
            covariance (np.array): 与该位姿相关的协方差矩阵

        返回:
            gtsam.Pose2: 添加了随机噪声的位姿
        """
        # 获取随机噪声并将其添加到提供的位姿
        delta = np.random.multivariate_normal(np.zeros(3), covariance)
        return pose.compose(n2g(delta, "Pose2"))

    def sample_current_pose(self) -> gtsam.Pose2:
        """使用 self.sample_pose() 将随机噪声添加到 self.current_keyframe.pose

        返回:
            gtsam.Pose2: 添加了随机噪声的 self.current_keyframe.pose
        """
        return self.sample_pose(self.current_keyframe.pose, self.current_keyframe.cov)

    def get_points(
        self, frames: list = None, ref_frame: Any = None, return_keys: bool = False
    ) -> np.array:
        """获取点云，执行以下步骤：
            - 累积帧中的点
            - 将它们转换为参考帧
            - 下采样点
            - 返回每个点的对应键

        参数:
            frames (list, 可选): 我们关心的帧的索引列表。默认为 None。
            ref_frame (Any, 可选): 我们想要相对于的帧，可以是 gtsam.Pose2 或 int 索引。默认为 None。
            return_keys (bool, 可选): 我们是否想要返回键？默认为 False。

        返回:
            np.array: 点云数组，可能带有每个点的键
        """
        # 如果没有指定帧，则获取所有帧
        if frames is None:
            frames = range(self.current_key)

        # 检查 ref_frame 是否为 gtsam.Pose2，如果不是，则假设它是 self.keyframes 列表中的索引
        if ref_frame is not None:
            if isinstance(ref_frame, gtsam.Pose2):
                ref_pose = ref_frame
            else:
                ref_pose = self.keyframes[ref_frame].pose

        # 定义一个空白数组来添加我们的点
        if return_keys:
            all_points = [np.zeros((0, 3), np.float32)]
        else:
            all_points = [np.zeros((0, 2), np.float32)]

        # 遍历提供的关键帧索引
        for key in frames:
            # 如果有一个参考帧，则使用该帧，否则使用 SLAM 帧
            if ref_frame is not None:
                # 将点转换为提供的参考帧
                points = self.keyframes[key].points
                pose = self.keyframes[key].pose
                transf = ref_pose.between(pose)
                transf_points = Keyframe.transform_points(points, transf)
            else:
                transf_points = self.keyframes[key].transf_points

            # 如果我们要每个点的键，在这里获取它们
            if return_keys:
                transf_points = np.c_[
                    transf_points, key * np.ones((len(transf_points), 1))
                ]
            all_points.append(transf_points)

        # 将点组合成一个 numpy 数组
        all_points = np.concatenate(all_points)

        # 应用体素下采样并返回
        if return_keys:
            return pcl.downsample(
                all_points[:, :2], all_points[:, (2,)], self.point_resolution
            )
        else:
            return pcl.downsample(all_points, self.point_resolution)

    def compute_icp(
        self,
        source_points: np.array,
        target_points: np.array,
        guess: np.array = gtsam.Pose2(),
    ) -> Union:
        """计算标准 ICP

        参数:
            source_points (np.array): 源点云 [x,y]
            target_points (np.array): 目标点云 [x,y]
            guess (np.array, 可选): 初始猜测，如果没有提供，则使用单位矩阵。默认为 gtsam.Pose2()。

        返回:
            Union[str,gtsam.Pose2]: 返回状态消息和结果作为 gtsam.Pose2
        """
        # 设置点
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # 将猜测转换为矩阵并应用 ICP
        guess = guess.matrix()
        message, T = self.icp.compute(source_points, target_points, guess)

        # 解析 ICP 输出
        x, y = T[:2, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])

        return message, gtsam.Pose2(x, y, theta)

    def compute_icp_with_cov(
        self, source_points: np.array, target_points: np.array, guesses: list
    ) -> Union:
        """计算具有协方差矩阵的 ICP

        参数:
            source_points (np.array): 源点云 [x,y]
            target_points (np.array): 目标点云 [x,y]
            guesses (list): 初始猜测列表

        返回:
            Union[str,gtsam.Pose2,np.array,np.array]: 状态消息、变换、协方差矩阵、测试的变换
        """
        # 解析点
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        # 检查每个提供的猜测与 ICP
        sample_transforms = []
        start = time_pkg.time()
        for g in guesses:
            g = g.matrix()
            message, T = self.icp.compute(source_points, target_points, g)

            # 只保留成功的情况
            if message == "success":
                x, y = T[:2, 2]
                theta = np.arctan2(T[1, 0], T[0, 0])
                sample_transforms.append((x, y, theta))

            # 强制最大运行时间
            if time_pkg.time() - start >= 2.0:
                break

        # 检查是否足够多的变换来获取协方差
        sample_transforms = np.array(sample_transforms)
        if len(sample_transforms) < 5:
            return "Too few samples for covariance computation", None, None, None

        # 不能使用 np.cov()。太多异常值
        try:
            fcov = MinCovDet(store_precision=False, support_fraction=0.8).fit(
                sample_transforms
            )
        except ValueError as e:
            return "Failed to calculate covariance", None, None, None

        # 解析结果
        m = n2g(fcov.location_, "Pose2")
        cov = fcov.covariance_

        # 旋转到局部帧
        R = m.rotation().matrix()
        cov[:2, :] = R.T.dot(cov[:2, :])
        cov[:, :2] = cov[:, :2].dot(R)

        # 检查默认的 ICP 协方差是否大于我们刚刚估计的协方差
        default_cov = np.diag(self.icp_odom_sigmas) ** 2
        if np.linalg.det(cov) < np.linalg.det(default_cov):
            cov = default_cov

        return "success", m, cov, sample_transforms

    def get_overlap(
        self,
        source_points: np.array,
        target_points: np.array,
        source_pose: gtsam.Pose2 = None,
        target_pose: gtsam.Pose2 = None,
        return_indices: bool = False,
    ) -> int:
        """获取提供的云之间的重叠，点数与最近邻

        参数:
            source_points (np.array): 源点云
            target_points (np.array): 目标点云
            source_pose (gtsam.Pose2, 可选): 源点的位姿。默认为 None。
            target_pose (gtsam.Pose2, 可选): 目标点的位姿。默认为 None。
            return_indices (bool, 可选): 如果我们要云的索引。默认为 False。

        返回:
            int: 具有最近邻的点数
        """
        # 如果我们要一个位姿，则转换点
        if source_pose:
            source_points = Keyframe.transform_points(source_points, source_pose)
        if target_pose:
            target_points = Keyframe.transform_points(target_points, target_pose)

        # 使用 PCL 匹配点使用最近邻
        # 注意未匹配的点在索引中得到 -1
        indices, dists = pcl.match(target_points, source_points, 1, self.point_noise)

        # 如果我们要索引，则发送那些
        if return_indices:
            return np.sum(indices != -1), indices
        else:
            return np.sum(indices != -1)

    def add_prior(self, keyframe: Keyframe) -> None:
        """为 SLAM 解决方案中的第一个位姿添加先验因子。这是起始帧。

        参数:
            keyframe (Keyframe): 初始帧的关键帧对象
        """
        pose = keyframe.pose
        factor = gtsam.PriorFactorPose2(X(0), pose, self.prior_model)
        self.graph.add(factor)
        self.values.insert(X(0), pose)

    def add_odometry(self, keyframe: Keyframe) -> None:
        """在提供的关键帧和最后一个关键帧之间添加里程计因子

        参数:
            keyframe (Keyframe): 传入的关键帧，基本上是 keyframe_t
        """
        # 获取提供的关键帧和最后一个记录的关键帧之间的时间差和位姿差
        dt = (keyframe.time - self.keyframes[-1].time).to_sec()
        dr_odom = self.keyframes[-1].pose.between(keyframe.pose)

        # 构建因子并将其插入图中，同时提供初始猜测
        factor = gtsam.BetweenFactorPose2(
            X(self.current_key - 1), X(self.current_key), dr_odom, self.odom_model
        )
        self.graph.add(factor)
        self.values.insert(X(self.current_key), keyframe.pose)

    def get_map(self, frames, resolution=None):
        # 在 slam_node 中实现
        # TODO 移除此代码
        raise NotImplementedError

    def get_matching_cost_subroutine1(
        self,
        source_points: np.array,
        source_pose: gtsam.Pose2,
        target_points: np.array,
        target_pose: gtsam.Pose2,
        source_pose_cov: np.array = None,
    ) -> Union:
        """执行全局成本点云对齐。这里我们将源点转换到目标点。

        参数:
            source_points (np.array): 源点云
            source_pose (gtsam.Pose2): source_points 的位姿
            target_points (np.array): 目标点云
            target_pose (gtsam.Pose2): target_points 的位姿
            source_pose_cov (np.array, 可选): 源点的协方差。默认为 None。

        返回:
            Union[function,list]: 要由 scipy.shgo 优化的函数和位姿列表
        """
        # pose_samples = []
        # target_tree = KDTree(target_points)

        # def subroutine(x):
        #     # x = [x, y, theta]
        #     delta = n2g(x, "Pose2")
        #     sample_source_pose = source_pose.compose(delta)
        #     sample_transform = target_pose.between(sample_source_pose)

        #     points = Keyframe.transform_points(source_points, sample_transform)
        #     dists, indices = target_tree.query(
        #         points, distance_upper_bound=self.point_noise
        #     )

        #     cost = -np.sum(indices != len(target_tree.data))

        #     pose_samples.append(np.r_[g2n(sample_source_pose), cost])
        #     return cost

        # return subroutine, pose_samples

        # maintain a list of poses we try
        pose_samples = []

        # 为目标点创建网格
        xmin, ymin = np.min(target_points, axis=0) - 2 * self.point_noise
        xmax, ymax = np.max(target_points, axis=0) + 2 * self.point_noise
        resolution = self.point_noise / 10.0
        xs = np.arange(xmin, xmax, resolution)
        ys = np.arange(ymin, ymax, resolution)
        target_grids = np.zeros((len(ys), len(xs)), np.uint8)

        # 为目标点填充网格
        r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
        r = np.clip(r, 0, target_grids.shape[0] - 1)
        c = np.clip(c, 0, target_grids.shape[1] - 1)
        target_grids[r, c] = 255

        # 膨胀网格
        dilate_hs = int(np.ceil(self.point_noise / resolution))
        dilate_size = 2 * dilate_hs + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
        )
        target_grids = cv2.dilate(target_grids, kernel)

        # # Calculate distance to the nearest points
        # target_grids = cv2.bitwise_not(target_grids)
        # target_grids = cv2.distanceTransform(target_grids, cv2.DIST_L2, 3)
        # target_grids = 1.0 - 0.2 * target_grids / self.point_noise
        # target_grids = np.clip(target_grids, 0.2, 1.0)

        source_pose_info = np.linalg.inv(source_pose_cov)

        def subroutine(x: np.array) -> float:
            """优化子程序，由 scipy.shgo 迭代调用

            参数:
                x (gtsam.Pose2): 作为数组的源位姿。 [x, y, theta]

            返回:
                float: 此步骤的成本
            """
            # 将传入的位姿打包为 gtsam.Pose2
            # 将此位姿应用于 source_pose 并获取源和目标之间的变换
            delta = n2g(x, "Pose2")
            sample_source_pose = source_pose.compose(delta)
            sample_transform = target_pose.between(sample_source_pose)

            # 将此新变换应用于源点
            # 然后将点限制为仅适合目标网格内的点
            points = Keyframe.transform_points(source_points, sample_transform)
            r = np.int32(np.round((points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((points[:, 0] - xmin) / resolution))
            inside = (
                (0 <= r)
                & (r < target_grids.shape[0])
                & (0 <= c)
                & (c < target_grids.shape[1])
            )

            # 获取重叠的单元格数量并记录位姿
            cost = -np.sum(target_grids[r[inside], c[inside]] > 0)
            pose_samples.append(np.r_[g2n(sample_source_pose), cost])

            return cost

        return subroutine, pose_samples

    def get_matching_cost_subroutine2(self, source_points, source_pose, occ):
        # TODO 移除此代码
        """
        Ceres 扫描匹配

        成本 = - sum_i  ||1 - M_nearest(Tx s_i)||^2,
                给定变换 Tx，源点 S，占据地图 M
        """
        pose_samples = []
        x0, y0, resolution, occ_arr = occ

        def subroutine(x):
            # x = [x, y, theta]
            delta = n2g(x, "Pose2")
            sample_pose = source_pose.compose(delta)

            xy = Keyframe.transform_points(source_points, sample_pose)
            r = np.int32(np.round((xy[:, 1] - y0) / resolution))
            c = np.int32(np.round((xy[:, 0] - x0) / resolution))

            sel = (r >= 0) & (c >= 0) & (r < occ_arr.shape[0]) & (c < occ_arr.shape[1])
            hit_probs_inside_map = occ_arr[r[sel], c[sel]]
            num_hits_outside_map = len(xy) - np.sum(sel)

            cost = (
                np.sum((1.0 - hit_probs_inside_map) ** 2)
                + num_hits_outside_map * (1.0 - 0.5) ** 2
            )
            cost = np.sqrt(cost / len(source_points))

            pose_samples.append(np.r_[g2n(sample_pose), cost])
            return cost

        return subroutine, pose_samples

    def initialize_sequential_scan_matching(
        self, keyframe: Keyframe
    ) -> InitializationResult:
        """通过使用全局 ICP 初始化顺序扫描匹配调用。

        参数:
            keyframe (Keyframe): 我们要注册的关键帧

        返回:
            InitializationResult: 初始化的结果
        """
        # 实例化一个 ICP InitializationResult 对象
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # 将当前关键帧与前 k 帧匹配
        ret.source_key = self.current_key
        ret.target_key = self.current_key - 1
        ret.source_pose = keyframe.pose
        ret.target_pose = self.current_keyframe.pose

        # 从前 k 帧（self.ssm_params.target_frames）累积参考点
        ret.source_points = keyframe.points
        target_frames = range(self.current_key)[-self.ssm_params.target_frames :]
        ret.target_points = self.get_points(target_frames, ret.target_key)
        ret.cov = np.diag(self.odom_sigmas)

        """if True:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        """if len(self.keyframes) % 2 == 0:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret"""

        # Only continue with this if it is enabled in slam.yaml
        if self.ssm_params.enable == False:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # check the source points for a minimum count
        if len(ret.source_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # 检查目标点是否达到最小数量
        if len(ret.target_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(ret.target_points))
            return ret

        # 检查我们是否已初始化 ICP 参数
        if not self.ssm_params.initialization:
            return ret

        with CodeTimer("SLAM - sequential scan matching - sampling"):
            # 定义 ICP 全局初始化的搜索空间
            pose_stds = np.array([self.odom_sigmas]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the global ICP subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # 使用 scipy.shgo 优化子程序
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.ssm_params.initialization_params[0],
                iters=self.ssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.ssm_params.initialization_params[2]}
                },
            )

        # 如果优化器指示成功，则打包结果返回
        if result.success:
            ret.source_pose_samples = np.array(pose_samples)
            ret.estimated_source_pose = ret.source_pose.compose(n2g(result.x, "Pose2"))
            ret.status.description = "matching cost {:.2f}".format(result.fun)

            # TODO 移除
            if self.save_data:
                ret.save("step-{}-ssm-sampling.npz".format(self.current_key))
        else:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message

        return ret

    def add_sequential_scan_matching(self, keyframe: Keyframe) -> None:
        """将顺序扫描匹配因子添加到图中。这里我们使用全局 ICP 作为标准 ICP 的初始猜测。
        然后我们执行一些简单的检查来捕获明显的异常值。如果这些检查通过，我们将 ICP 结果添加到位姿图中。

        参数:
            keyframe (Keyframe): 我们正在评估的关键帧，这包含所有相关信息。
        """
        # 调用全局 ICP
        ret = self.initialize_sequential_scan_matching(keyframe)

        # TODO 移除
        if self.save_fig:
            ret.plot("step-{}-ssm-sampling.png".format(self.current_key))

        # 检查全局 ICP 调用的状态，如果结果是失败。
        # 只需添加里程计因子并返回
        if not ret.status:
            self.add_odometry(keyframe)
            return

        # 将全局 ICP 复制到 ICPResult
        ret2 = ICPResult(ret, self.ssm_params.cov_samples > 0)

        # 在这里用计时器计算 ICP
        with CodeTimer("SLAM - sequential scan matching - ICP"):
            # 如果可能，使用协方差估计计算 ICP
            if self.ssm_params.initialization and self.ssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.ssm_params.cov_samples],
                )

                # 如果 ICP 失败，将其推入 ret2 对象
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                # 否则将 ICP 信息推入 ret2
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # 否则调用标准 ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # 检查失败
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.status.description = ""

        # 与航位推算相比，变换不能太大
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.ssm_params.max_translation
                or delta_rotation > self.ssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # 两个点云之间必须有足够的重叠。
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points, ret2.estimated_transform
            )
            if overlap < self.ssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = "overlap {}".format(overlap)

        if ret2.status:
            # 如果我们使用带协方差的 ICP，则不需要样板噪声模型
            if ret2.cov is not None:
                icp_odom_model = self.create_full_noise_model(ret2.cov)
            else:
                icp_odom_model = self.icp_odom_model

            # 打包要添加到图中的因子
            factor = gtsam.BetweenFactorPose2(
                X(ret2.target_key),
                X(ret2.source_key),
                ret2.estimated_transform,
                icp_odom_model,
            )

            # 添加因子和这个新位姿的初始猜测
            self.graph.add(factor)
            self.values.insert(
                X(ret2.source_key), ret2.target_pose.compose(ret2.estimated_transform)
            )
            ret2.inserted = True  # 记录为已添加

            # TODO 移除
            if self.save_data:
                ret2.save("step-{}-ssm-icp.npz".format(self.current_key))

        # 如果 ICP 失败，则只推入航位推算信息
        else:
            self.add_odometry(keyframe)

        # TODO 移除
        if self.save_fig:
            ret2.plot("step-{}-ssm-icp.png".format(self.current_key))

    def initialize_nonsequential_scan_matching(self) -> InitializationResult:
        """初始化非顺序扫描匹配调用。这里我们使用全局 ICP 来检查最近的关键帧与地图其余部分的回环。

        返回:
            InitializationResult: 全局 ICP 结果
        """
        # 实例化一个对象来捕获结果
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        # 获取我们关心的回环检测搜索的索引
        ret.source_key = self.current_key - 1
        ret.source_pose = self.current_frame.pose
        ret.estimated_source_pose = ret.source_pose
        # 聚合源云，这里我们想要 k 帧（self.nssm_params.source_frames）
        source_frames = range(
            ret.source_key, ret.source_key - self.nssm_params.source_frames, -1
        )
        ret.source_points = self.get_points(source_frames, ret.source_key)

        # 将回环检测搜索限制为具有足够点的那些
        if len(ret.source_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # 查找用于匹配的目标点
        # 限制搜索关键帧。这里我们想要所有关键帧减去 k（self.nssm_params.min_st_sep）
        target_frames = range(self.current_key - self.nssm_params.min_st_sep)

        # 全局坐标系中的目标点
        target_points, target_keys = self.get_points(target_frames, None, True)

        # 遍历源帧
        # 消除不在同一视场中的帧
        sel = np.zeros(len(target_points), np.bool)
        for source_frame in source_frames:
            # 提取位姿和协方差信息
            pose = self.keyframes[source_frame].pose
            cov = self.keyframes[source_frame].cov

            # 解析协方差
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            range_bound = translation_std * 5.0 + self.oculus.max_range
            bearing_bound = rotation_std * 5.0 + self.oculus.horizontal_aperture * 0.5

            # 找出不确定的点
            local_points = Keyframe.transform_points(target_points, pose.inverse())
            ranges = np.linalg.norm(local_points, axis=1)
            bearings = np.arctan2(local_points[:, 1], local_points[:, 0])
            sel_i = (ranges < range_bound) & (abs(bearings) < bearing_bound)
            sel |= sel_i

        # 只保留确定的点
        target_points = target_points[sel]
        target_keys = target_keys[sel]

        # 检查哪个帧在附近有最多的点
        target_frames, counts = np.unique(np.int32(target_keys), return_counts=True)
        target_frames = target_frames[counts > 10]
        counts = counts[counts > 10]

        # 检查聚合云的点数
        if len(target_frames) == 0 or len(target_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(target_points))
            return ret

        # 用一些信息填充初始化对象
        ret.target_key = target_frames[
            np.argmax(counts)
        ]  # 这是关键的，具有最多重叠点的那个
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = Keyframe.transform_points(
            target_points, ret.target_pose.inverse()
        )
        ret.cov = self.keyframes[ret.source_key].cov

        # 检查我们是否有全局 ICP 的参数
        if not self.nssm_params.initialization:
            return ret

        with CodeTimer("SLAM - nonsequential scan matching - sampling"):
            # 设置全局 ICP 的边界
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            pose_stds = np.array([[translation_std, translation_std, rotation_std]]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # TODO remove
            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            # build the subroutine
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            # 使用 scipy.shgo 优化
            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.nssm_params.initialization_params[0],
                iters=self.nssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.nssm_params.initialization_params[2]}
                },
            )

        # 检查 shgo 结果
        if not result.success:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message
            return ret

        # 解析结果
        delta = n2g(result.x, "Pose2")
        ret.estimated_source_pose = ret.source_pose.compose(delta)
        ret.source_pose_samples = np.array(pose_samples)
        ret.status.description = "matching cost {:.2f}".format(result.fun)

        # Refine target key by searching for the pose with maximum overlap
        # with current source points
        estimated_source_points = Keyframe.transform_points(
            ret.source_points, ret.estimated_source_pose
        )
        overlap, indices = self.get_overlap(
            estimated_source_points, target_points, return_indices=True
        )
        target_frames1, counts1 = np.unique(
            np.int32(target_keys[indices[indices != -1]]), return_counts=True
        )
        if len(counts1) == 0:
            ret.status = STATUS.NOT_ENOUGH_OVERLAP
            ret.status.description = "0"
            return ret

        # TODO 移除
        if self.save_data:
            ret.save("step-{}-nssm-sampling.npz".format(self.current_key - 1))

        # 记录目标键和
        # 使用新的目标键在目标帧中重新计算目标点
        ret.target_key = target_frames1[np.argmax(counts1)]
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = self.get_points(target_frames, ret.target_key)

        return ret

    def add_nonsequential_scan_matching(self) -> ICPResult:
        """运行回环检测搜索。这里我们将最近的关键帧与之前的帧进行比较。
        如果找到回环，则通过 PCM 进行几何验证。

        返回:
            ICPResult: 我们找到的回环，返回用于调试目的
        """
        # 如果我们没有足够的关键帧来聚合子图，则返回
        if self.current_key < self.nssm_params.min_st_sep:
            return

        # 使用全局 ICP 调用初始化搜索
        ret = self.initialize_nonsequential_scan_matching()

        # 如果全局 ICP 调用不起作用，则返回
        if not ret.status:
            return

        # 打包全局 ICP 调用结果
        ret2 = ICPResult(ret, self.nssm_params.cov_samples > 0)

        # 在这里用计时器计算 ICP
        with CodeTimer("SLAM - nonsequential scan matching - ICP"):
            # 如果可能，使用协方差矩阵计算 ICP
            if self.nssm_params.initialization and self.nssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.nssm_params.cov_samples],
                )

                # 检查状态
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )

            # 否则使用标准 ICP
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )

                # 检查状态
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret.status.description = ""

        # 添加一些失败检测
        # 与初始猜测相比，变换不能太大
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = np.linalg.norm(delta.translation())
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.nssm_params.max_translation
                or delta_rotation > self.nssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # 两个点云之间必须有足够的重叠。
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points[:, :2], ret2.estimated_transform
            )
            if overlap < self.nssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = str(overlap)
            
        # 应用几何验证，在这种情况下是 PCM
        if ret2.status:
            # 更新 pcm 队列
            while (
                self.nssm_queue
                and ret2.source_key - self.nssm_queue[0].source_key
                > self.pcm_queue_size
            ):
                self.nssm_queue.pop(0)

            # 将最新的回环记录到 pcm 队列中并检查 PCM
            self.nssm_queue.append(ret2)
            pcm = self.verify_pcm(self.nssm_queue,self.min_pcm)

            # 如果 PCM 结果没有回环，列表 pcm 将为空
            # 遍历任何结果并将它们添加到图中
            for m in pcm:
                # 从 pcm 队列中提取回环
                ret2 = self.nssm_queue[m]

                # 检查回环是否已添加到图中
                if not ret2.inserted:
                    # 获取噪声模型
                    if ret2.cov is not None:
                        icp_odom_model = self.create_full_noise_model(ret2.cov)
                    else:
                        icp_odom_model = self.icp_odom_model

                    # 构建因子并将其添加到图中
                    factor = gtsam.BetweenFactorPose2(
                        X(ret2.target_key),
                        X(ret2.source_key),
                        ret2.estimated_transform,
                        icp_odom_model,
                    )
                    self.graph.add(factor)
                    self.keyframes[ret2.source_key].constraints.append(
                        (ret2.target_key, ret2.estimated_transform)
                    )
                    ret2.inserted = True  # 更新此回环的状态，不要添加两次回环

        return ret2

    def is_keyframe(self, frame: Keyframe) -> bool:
        """检查 Keyframe 对象是否满足成为 SLAM 关键帧的条件。
        如果车辆移动足够多。无论是旋转还是平移。

        参数:
            frame (Keyframe): 我们要检查的关键帧。

        返回:
            bool: 指示我们是否需要将此帧添加到 SLAM 解决方案的标志
        """
        # 如果我们的 SLAM 解决方案中没有关键帧，这是第一个
        if not self.keyframes:
            return True

        # 检查时间
        duration = frame.time - self.current_keyframe.time
        if duration < self.keyframe_duration:
            return False

        # 检查旋转和平移
        dr_odom = self.keyframes[-1].dr_pose.between(frame.dr_pose)
        translation = np.linalg.norm(dr_odom.translation())
        rotation = abs(dr_odom.theta())

        return (
            translation > self.keyframe_translation or rotation > self.keyframe_rotation
        )

    def create_full_noise_model(
        self, cov: np.array
    ) -> gtsam.noiseModel.Gaussian.Covariance:
        """使用 gtsam api 从 numpy 数组创建噪声模型。

        参数:
            cov (np.array): 协方差矩阵的 numpy 数组。

        返回:
            gtsam.noiseModel.Gaussian.Covariance: 输入的 gtsam 版本
        """
        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def create_robust_full_noise_model(self, cov: np.array) -> gtsam.noiseModel.Robust:
        """从 numpy 数组创建稳健的 gtsam 噪声模型

        参数:
            cov (np.array): 协方差矩阵的 numpy 数组

        返回:
            gtsam.noiseModel.Robust: 输入的 gtsam 版本
        """
        model = gtsam.noiseModel.Gaussian.Covariance(cov)
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def create_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Diagonal:
        """从 sigma 列表创建噪声模型，视为对角矩阵。

        返回:
            gtsam.noiseModel.Diagonal: 输入的 gtsam 版本
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])

    def create_robust_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Robust:
        """从 sigma 列表创建稳健的噪声模型

        返回:
            gtsam.noiseModel.Robust: 输入的 gtsam 版本
        """
        model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def update_factor_graph(self, keyframe: Keyframe = None) -> None:
        """更新内部 SLAM 估计

        参数:
            keyframe (Keyframe, 可选): 需要添加到 SLAM 解决方案的关键帧。默认为 None。
        """
        # 如果我们有关键帧，将其添加到我们的关键帧列表中
        if keyframe:
            self.keyframes.append(keyframe)

        # 将最新的因子推送到 ISAM2 实例
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)  # 一旦我们将其推送到 ISAM2，就清除图和值
        self.values.clear()

        # 更新整个轨迹
        values = self.isam.calculateEstimate()
        for x in range(values.size()):
            pose = values.atPose2(X(x))
            self.keyframes[x].update(pose)

        # 只更新最新的协方差
        cov = self.isam.marginalCovariance(X(values.size() - 1))
        self.keyframes[-1].update(pose, cov)

        # 更新待处理回环中的位姿以进行 PCM
        for ret in self.nssm_queue:
            ret.source_pose = self.keyframes[ret.source_key].pose
            ret.target_pose = self.keyframes[ret.target_key].pose
            if ret.inserted:
                ret.estimated_transform = ret.target_pose.between(ret.source_pose)

    def verify_pcm(self, queue: list, min_pcm_value: int) -> list:
        """获取成对一致的测量。

        参数:
            queue (list): 正在检查的回环列表。
            min_pcm_value (int): 我们想要的最小 pcm 值

        返回:
            list: 返回任何成对一致的回环。我们返回提供的队列中的索引列表。
        """
        # 检查我们是否有足够的回环来关注
        if len(queue) < min_pcm_value:
            return []

        # 将回环转换为一致性图
        G = defaultdict(list)
        for (a, ret_il), (b, ret_jk) in combinations(zip(range(len(queue)), queue), 2):
            pi = ret_il.target_pose
            pj = ret_jk.target_pose
            pil = ret_il.estimated_transform
            plk = ret_il.source_pose.between(ret_jk.source_pose)
            pjk1 = ret_jk.estimated_transform
            pjk2 = pj.between(pi.compose(pil).compose(plk))

            error = gtsam.Pose2.Logmap(pjk1.between(pjk2))
            md = error.dot(np.linalg.inv(ret_jk.cov)).dot(error)
            # chi2.ppf(0.99, 3) = 11.34
            if md < 11.34:  # 这不是一个魔法数字
                G[a].append(b)
                G[b].append(a)

        # 找到一致回环的集合
        maximal_cliques = list(self.find_cliques(G))

        # 如果我们没有得到任何东西，返回空
        if not maximal_cliques:
            return []

        # 排序并只返回最大的集合，同时检查集合是否足够大
        maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
        if len(maximum_clique) < min_pcm_value:
            return []

        return maximum_clique

    def find_cliques(self, G: defaultdict):
        """返回无向图中的所有最大团。

        参数:
            G (defaultdict): 一致性图
        """
        if len(G) == 0:
            return

        adj = {u: {v for v in G[u] if v != u} for u in G}
        Q = [None]

        subg = set(G)
        cand = set(G)
        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]
        stack = []

        try:
            while True:
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    adj_q = adj[q]
                    subg_q = subg & adj_q
                    if not subg_q:
                        yield Q[:]
                    else:
                        cand_q = cand & adj_q
                        if cand_q:
                            stack.append((subg, cand, ext_u))
                            Q.append(None)
                            subg = subg_q
                            cand = cand_q
                            u = max(subg, key=lambda u: len(cand & adj[u]))
                            ext_u = cand - adj[u]
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass
