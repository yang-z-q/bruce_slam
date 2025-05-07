import numpy as np
import cv2
from scipy.special import logit, expit
from nav_msgs.msg import OccupancyGrid

from .sonar import *
from .utils.conversions import *
from . import pcl


class Submap(object):
    def __init__(self):
        # 索引
        self.k = 0

        # gtsam.Pose2
        self.pose = None

        # 声纳坐标系中每个像素的 x, y 坐标（float32）
        # 如果为 None，使用前一个
        self.sonar_xy = None

        # 占据栅格地图（float32）
        self.logodds = None
        # 强度栅格地图（uint8）
        self.intensity = None

        # 缓存上次更新的索引（uint16）
        self.r, self.c = None, None
        # 和 logodds（float32）
        self.l = None
        # 和强度（uint32）
        self.i = None

        #############################################
        # 用于绘图
        #############################################
        self.cimg = None
        self.limg = None


class Mapping(object):
    def __init__(self):
        #  (x0, y0) ----> x
        #    |                   转换到
        #    |                <=================  声纳图像（方位角，距离）
        #    V                   全局平面
        #    y

        # 地图大小
        self.x0 = -50.0
        self.y0 = -50.0
        self.width = 100.0
        self.height = 100.0
        # 通过增加 50.0m 动态调整边界
        self.inc = 50.0
        self.resolution = 0.2
        self.rows = None
        self.cols = None

        self.oculus = OculusProperty()
        self.oculus_image_size = None
        self.oculus_r_skip = None
        self.oculus_c_skip = None

        # 栅格单元中的累积强度
        self.intensity_grid = None
        # 栅格单元中的观测计数
        self.counter_grid = None

        # 占据栅格地图
        ###################################
        # 方法 1：使用 logodds 更新规则
        ###################################
        # 强度栅格地图
        self.pub_intensity = False
        # 为了在闭环期间更新其中一个关键帧，
        # 不能使用钳制更新策略。
        self.pub_occupancy1 = True
        self.hit_prob = 0.8
        self.miss_prob = 0.3
        self.logodds_grid = None
        self.inflation_angle = 0.05
        self.inflation_range = 0.5
        ###################################
        # 方法 2：使用点投影
        ###################################
        self.pub_occupancy2 = True
        self.point_cloud = None
        self.inflation_radius = 0.5
        ###################################

        # 在构建占据地图之前移除烦人的离群点
        self.outlier_filter_radius = 5.0
        self.outlier_filter_min_points = 20

        # 只更新有显著运动的关键帧
        self.min_translation = 0.5
        self.min_rotation = 0.05

        # 跟踪已编辑的边界框
        # 只发布框内的地图
        self.rmin, self.rmax = None, None
        self.cmin, self.cmax = None, None

        # 位姿，声纳数据
        self.keyframes = []
        self.point_cloud = None

        self.save_fig = False

    def configure(self):
        self.hit_logodds = logit(self.hit_prob)
        self.miss_logodds = logit(self.miss_prob)

        xs = np.arange(0, self.width, self.resolution)
        ys = np.arange(0, self.height, self.resolution)
        self.rows = len(ys)
        self.cols = len(xs)

        if self.pub_occupancy1:
            self.logodds_grid = np.zeros((ys.shape[0], xs.shape[0]), np.float32)

        if self.pub_occupancy2:
            dilate_hs = int(np.ceil(self.inflation_radius / self.resolution))
            self.dilate_size = 2 * dilate_hs + 1

        if self.pub_intensity:
            self.intensity_grid = np.zeros((ys.shape[0], xs.shape[0]), np.uint32)
            self.counter_grid = np.zeros_like(self.intensity_grid, np.uint16)

        self.rmax = self.cmax = 0
        self.rmin = ys.shape[0] - 1
        self.cmin = xs.shape[0] - 1
        self.inc_r = int(self.inc / self.resolution)
        self.inc_c = int(self.inc / self.resolution)

    def pose_changed(self, pose, new_pose):
        dp = pose.between(new_pose)
        dt = np.linalg.norm(dp.translation())
        dr = abs(dp.theta())

        return dt > self.min_translation or dr > self.min_rotation

    def add_keyframe(self, key, pose, ping, points):
        changed = self.oculus.configure(ping)

        keyframe = Submap()
        keyframe.k = len(self.keyframes)
        keyframe.pose = pose

        if changed:
            # 下采样原始图像
            self.oculus_r_skip = max(
                1, np.int32(np.floor(self.resolution / self.oculus.range_resolution))
            )
            range_resolution = self.oculus.angular_resolution * self.oculus.max_range
            self.oculus_c_skip = max(
                1, np.int32(np.floor(self.resolution / range_resolution))
            )

            B, R = np.meshgrid(
                self.oculus.bearings[:: self.oculus_c_skip],
                self.oculus.ranges[:: self.oculus_r_skip],
            )
            X, Y = np.cos(B) * R, np.sin(B) * R
            keyframe.sonar_xy = np.c_[X.ravel(), Y.ravel()].astype(np.float32)
            self.oculus_image_size = X.shape

        if self.pub_occupancy1:
            # mask = np.zeros(self.oculus_image_size, np.uint8)
            mask = np.zeros(self.oculus_image_size, np.float32)

            if len(points):

                if self.outlier_filter_min_points > 1:
                    points = pcl.remove_outlier(
                        points[:, :2],
                        self.outlier_filter_radius,
                        self.outlier_filter_min_points,
                    )

                c = self.oculus.b2c(np.arctan2(points[:, 1], points[:, 0]))
                c = np.clip(np.int32(np.round(c)), 0, self.oculus.num_bearings - 1)
                r = self.oculus.ra2ro(np.linalg.norm(points[:, :2], axis=1))
                r = np.clip(np.int32(np.round(r)), 0, self.oculus.num_ranges - 1)
                mask[r // self.oculus_r_skip, c // self.oculus_c_skip] = 1.0

                hc = int(
                    round(
                        self.inflation_angle
                        / self.oculus.angular_resolution
                        / self.oculus_c_skip
                    )
                )
                hr = int(
                    round(
                        self.inflation_range
                        / self.oculus.range_resolution
                        / self.oculus_r_skip
                    )
                )

                # kernel = cv2.getStructuringElement(
                #     cv2.MORPH_ELLIPSE, (hc * 2 + 1, hr * 2 + 1), (hc, hr)
                # )
                # mask = cv2.dilate(mask, kernel)

                kernel_r = cv2.getGaussianKernel(2 * hr + 1, -1)
                kernel_c = cv2.getGaussianKernel(2 * hc + 1, -1)
                kernel = kernel_r.dot(kernel_c.T)
                mask = cv2.filter2D(
                    mask, cv2.CV_32F, kernel, None, None, 0.0, cv2.BORDER_CONSTANT
                )
                mask /= kernel[hr, hc] / self.hit_prob
                mask = np.clip(mask, 0.5, self.hit_prob)

                # 只将第一次命中之前的点标记为未命中
                first_hits = np.argmax(mask > 0.5, axis=0)
                # 如果没有命中，将所有点标记为未命中
                first_hits[first_hits == 0] = mask.shape[0]
                for j in range(mask.shape[1]):
                    mask[: first_hits[j], j] = self.miss_prob
            else:
                mask += self.miss_prob

            logodds = logit(mask)
            keyframe.logodds = logodds.ravel().astype(np.float32)

            #############################################
            # 保存一些图像用于绘图
            #############################################
            if self.save_fig:
                keyframe.cimg = r2n(ping)
                keyframe.limg = logodds
            #############################################

        if self.pub_occupancy2:
            self.point_cloud = points

        if self.pub_intensity:
            intensity = r2n(ping.ping)[::r_skip, ::c_skip]
            keyframe.intensity = intensity.ravel()

        self.fit_grid(keyframe)
        self.inc_grid(keyframe)

        # 以防我们错过一个关键帧
        while len(self.keyframes) < key:
            self.keyframes.append(None)

        self.keyframes.append(keyframe)

    def update_pose(self, key, new_pose):
        assert key < len(self.keyframes)
        keyframe = self.keyframes[key]
        if not keyframe:
            return

        pose = keyframe.pose
        if not self.pose_changed(pose, new_pose):
            return

        keyframe.pose = new_pose
        # Remove old measurements
        self.dec_grid(keyframe)
        # Transform new measurements to global frame
        self.fit_grid(keyframe)
        # Add new measurements
        self.inc_grid(keyframe)

    def get_intensity_grid(self):
        occ_msg = OccupancyGrid()
        occ_msg.header.frame_id = "map"

        # Only publish updated box
        intensity = self.intensity_grid[
            self.rmin : self.rmax + 1, self.cmin : self.cmax + 1
        ]
        counter = self.counter_grid[
            self.rmin : self.rmax + 1, self.cmin : self.cmax + 1
        ]
        occ = np.ones_like(intensity, np.int8) * -1
        sel = counter > 0
        occ[sel] = np.int8(np.round(intensity[sel] / 255.0 * 100.0 / counter[sel]))

        occ_msg.info.origin.position.x = self.x0 + self.cmin * self.resolution
        occ_msg.info.origin.position.y = self.y0 + self.rmin * self.resolution
        occ_msg.info.origin.orientation.x = 0
        occ_msg.info.origin.orientation.y = 0
        occ_msg.info.origin.orientation.z = 0
        occ_msg.info.origin.orientation.w = 1
        occ_msg.info.width = self.cmax - self.cmin + 1
        occ_msg.info.height = self.rmax - self.rmin + 1
        occ_msg.info.resolution = self.resolution
        occ_msg.data = list(occ.ravel())

        return occ_msg

    def get_occupancy_grid(self, frames=None, resolution=None):
        if self.pub_occupancy1:
            return self.get_occupancy_grid1(frames, resolution)
        elif self.pub_occupancy2:
            return self.get_occupancy_grid2(frames, resolution)

    def get_occupancy_grid1(self, frames=None, resolution=None):
        occ_msg = OccupancyGrid()
        occ_msg.header.frame_id = "map"

        if frames is None:
            logodds_grid = self.logodds_grid
            rmin, rmax, cmin, cmax = self.rmin, self.rmax, self.cmin, self.cmax
        else:
            logodds_grid = np.zeros_like(self.logodds_grid)
            rmin, rmax, cmin, cmax = self.rmax, self.rmin, self.cmax, self.cmin
            for k in frames:
                if k >= len(self.keyframes) or self.keyframes[k] is None:
                    continue
                keyframe = self.keyframes[k]
                logodds_grid[keyframe.r, keyframe.c] += keyframe.l
                rmin = min(rmin, keyframe.r.min())
                rmax = max(rmax, keyframe.r.max())
                cmin = min(cmin, keyframe.c.min())
                cmax = max(cmax, keyframe.c.max())

        # Only publish updated box
        logodds = logodds_grid[rmin : rmax + 1, cmin : cmax + 1]
        probs = expit(logodds)

        if (
            resolution is not None
            and resolution > 0
            and abs(resolution - self.resolution) > self.resolution * 1e-1
        ):
            assert resolution >= self.resolution
            ratio = self.resolution / resolution

            probs = cv2.resize(probs, None, None, ratio, ratio, cv2.INTER_NEAREST)
            resolution = self.resolution / ratio
        else:
            resolution = self.resolution

        occ = np.int8(np.clip(100 * probs, 0, 100))
        occ_msg.info.origin.position.x = self.x0 + cmin * resolution
        occ_msg.info.origin.position.y = self.y0 + rmin * resolution
        occ_msg.info.origin.orientation.x = 0
        occ_msg.info.origin.orientation.y = 0
        occ_msg.info.origin.orientation.z = 0
        occ_msg.info.origin.orientation.w = 1
        occ_msg.info.width = occ.shape[1]
        occ_msg.info.height = occ.shape[0]
        occ_msg.info.resolution = resolution
        occ_msg.data = list(occ.ravel())

        return occ_msg

    def get_occupancy_grid2(self, frames=None, resolution=None):
        occ_msg = OccupancyGrid()
        occ_msg.header.frame_id = "map"

        # Default unknown
        size = self.rmax - self.rmin + 1, self.cmax - self.cmin + 1
        occ = np.ones(size, np.int8) * -1

        points = self.point_cloud[:, :2]
        if frames is not None:
            points = [np.zeros((0, 2))]
            keys = np.uint32(self.point_cloud[:, 3])
            for k in frames:
                frame_points = self.point_cloud[keys == k, :2]
                points.append(frame_points)
            points = np.concatenate(points)

        # Observed as free
        if frames is None:
            frames = range(len(self.keyframes))
        for k in frames:
            if k >= len(self.keyframes) or self.keyframes[k] is None:
                continue
            keyframe = self.keyframes[k]
            occ[keyframe.r - self.rmin, keyframe.c - self.cmin] = 0

        # Publish known region
        rmin = np.nonzero(np.max(occ, axis=1) != -1)[0][0]
        cmin = np.nonzero(np.max(occ, axis=0) != -1)[0][0]
        rmax = np.nonzero(np.max(occ, axis=1) != -1)[0][-1]
        cmax = np.nonzero(np.max(occ, axis=0) != -1)[0][-1]
        occ = occ[rmin : rmax + 1, cmin : cmax + 1]
        rmin += self.rmin
        cmin += self.cmin

        # Remove outliers that have few points within radius
        if self.outlier_filter_min_points > 1:
            points = pcl.remove_outlier(
                points[:, :2],
                self.outlier_filter_radius,
                self.outlier_filter_min_points,
            )

        # Projected points to occupied
        x0 = self.x0 + cmin * self.resolution
        y0 = self.y0 + rmin * self.resolution
        r = np.int32(np.round((points[:, 1] - y0) / self.resolution))
        c = np.int32(np.round((points[:, 0] - x0) / self.resolution))
        sel = (0 <= r) & (r < occ.shape[0]) & (0 <= c) & (c < occ.shape[1])
        r, c = r[sel], c[sel]
        mask = np.zeros_like(occ, np.uint8)
        mask[r, c] = 255

        # Inflate occupied cells
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_size,) * 2)
        mask = cv2.dilate(mask, kernel)
        occ[mask > 0] = 100

        if (
            resolution is not None
            and resolution > 0
            and abs(resolution - self.resolution) > self.resolution * 1e-1
        ):
            assert resolution >= self.resolution
            ratio = self.resolution / resolution

            occ = cv2.resize(occ, None, None, ratio, ratio, cv2.INTER_NEAREST)
            resolution = self.resolution / ratio
        else:
            resolution = self.resolution

        occ_msg.info.origin.position.x = x0
        occ_msg.info.origin.position.y = y0
        occ_msg.info.origin.orientation.x = 0
        occ_msg.info.origin.orientation.y = 0
        occ_msg.info.origin.orientation.z = 0
        occ_msg.info.origin.orientation.w = 1
        occ_msg.info.width = occ.shape[1]
        occ_msg.info.height = occ.shape[0]
        occ_msg.info.resolution = resolution
        occ_msg.data = list(occ.ravel())

        return occ_msg

    def inc_grid(self, keyframe):
        if self.pub_intensity:
            self.intensity_grid[keyframe.r, keyframe.c] += keyframe.i
            self.counter_grid[keyframe.r, keyframe.c] += 1

        if self.pub_occupancy1:
            self.logodds_grid[keyframe.r, keyframe.c] += keyframe.l

        if len(keyframe.r):
            self.rmin = min(self.rmin, keyframe.r.min())
            self.rmax = max(self.rmax, keyframe.r.max())
        if len(keyframe.c):
            self.cmin = min(self.cmin, keyframe.c.min())
            self.cmax = max(self.cmax, keyframe.c.max())

    def dec_grid(self, keyframe):
        if self.pub_intensity:
            self.intensity_grid[keyframe.r, keyframe.c] -= keyframe.i
            self.counter_grid[keyframe.r, keyframe.c] -= 1

        if self.pub_occupancy1:
            self.logodds_grid[keyframe.r, keyframe.c] -= keyframe.l

        # Boundary never decreases

    def fit_grid(self, keyframe):
        yaw = keyframe.pose.theta()
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        t = np.array([keyframe.pose.x(), keyframe.pose.y()])

        # Calculate x, y coordinates for every pixel in sonar frame
        sonar_xy = keyframe.sonar_xy
        if sonar_xy is None:
            k = keyframe.k - 1
            while k >= 0:
                if self.keyframes[k] and self.keyframes[k].sonar_xy is not None:
                    sonar_xy = self.keyframes[k].sonar_xy
                    break
                k -= 1
            assert sonar_xy is not None
        xy = R.dot(sonar_xy.T).T + t

        r = np.int32(np.round((xy[:, 1] - self.y0) / self.resolution))
        c = np.int32(np.round((xy[:, 0] - self.x0) / self.resolution))
        r, c = self.adjust_bounds(r, c)

        # Remove duplicate indices
        idx = r * self.cols + c
        _, sel = np.unique(idx, return_index=True)

        keyframe.r = np.uint16(r[sel])
        keyframe.c = np.uint16(c[sel])

        if self.pub_occupancy1:
            keyframe.l = keyframe.logodds[sel]

        if self.pub_intensity:
            keyframe.i = keyframe.intensity[sel]

    def adjust_bounds(self, r, c):
        while not np.all(r >= 0):
            r += self.inc_r
            self.rmin += self.inc_r
            self.rmax += self.inc_r
            self.rows += self.inc_r
            self.y0 -= self.inc_r * self.resolution
            self.height += self.inc_r * self.resolution
            if self.pub_occupancy1:
                self.logodds_grid = np.r_[
                    np.zeros((self.inc_r, self.cols), self.logodds_grid.dtype),
                    self.logodds_grid,
                ]
            if self.pub_intensity:
                self.intensity_grid = np.r_[
                    np.zeros((self.inc_r, self.cols), self.intensity_grid.dtype),
                    self.intensity_grid,
                ]
                self.counter_grid = np.r_[
                    np.zeros((self.inc_r, self.cols), self.counter_grid.dtype),
                    self.counter_grid,
                ]
            for keyframe in self.keyframes:
                keyframe.r += self.inc_r
        while not np.all(r < self.rows):
            self.rows += self.inc_r
            self.height += self.inc_r * self.resolution
            if self.pub_occupancy1:
                self.logodds_grid = np.r_[
                    self.logodds_grid,
                    np.zeros((self.inc_r, self.cols), self.logodds_grid.dtype),
                ]
            if self.pub_intensity:
                self.intensity_grid = np.r_[
                    self.intensity_grid,
                    np.zeros((self.inc_r, self.cols), self.intensity_grid.dtype),
                ]
                self.counter_grid = np.r_[
                    self.counter_grid,
                    np.zeros((self.inc_r, self.cols), self.counter_grid.dtype),
                ]
        while not np.all(c >= 0):
            c += self.inc_c
            self.cmin += self.inc_c
            self.cmax += self.inc_c
            self.cols += self.inc_c
            self.x0 -= self.inc_c * self.resolution
            self.width += self.inc_c * self.resolution
            if self.pub_occupancy1:
                self.logodds_grid = np.c_[
                    np.zeros((self.rows, self.inc_c), self.logodds_grid.dtype),
                    self.logodds_grid,
                ]
            if self.pub_intensity:
                self.intensity_grid = np.r_[
                    np.zeros((self.rows, self.inc_r), self.intensity_grid.dtype),
                    self.intensity_grid,
                ]
                self.counter_grid = np.r_[
                    np.zeros((self.rows, self.inc_c), self.counter_grid.dtype),
                    self.counter_grid,
                ]
            for keyframe in self.keyframes:
                keyframe.c += self.inc_c
        while not np.all(c < self.cols):
            self.cols += self.inc_c
            self.width += self.inc_c * self.resolution
            if self.pub_occupancy1:
                self.logodds_grid = np.c_[
                    self.logodds_grid,
                    np.zeros((self.rows, self.inc_c), self.logodds_grid.dtype),
                ]
            if self.pub_intensity:
                self.intensity_grid = np.r_[
                    self.intensity_grid,
                    np.zeros((self.rows, self.inc_r), self.intensity_grid.dtype),
                ]
                self.counter_grid = np.r_[
                    self.counter_grid,
                    np.zeros((self.rows, self.inc_c), self.counter_grid.dtype),
                ]
        return r, c
