import math
import numpy as np
from scipy.optimize import root

from .utils import * 
from .sonar import *
from . import cfar

class CFAR(object):
    """
    恒虚警率(CFAR)检测的几种变体
        - 单元平均(CA)CFAR
        - 最大单元平均(GOCA)CFAR
        - 有序统计(OS)CFAR
    """

    def __init__(self, Ntc, Ngc, Pfa, rank=None):
        self.Ntc = Ntc # 训练单元数量
        assert self.Ntc % 2 == 0
        self.Ngc = Ngc # 保护单元数量
        assert self.Ngc % 2 == 0
        self.Pfa = Pfa # 虚警率
        if rank is None: # 矩阵秩
            self.rank = self.Ntc / 2
        else:
            self.rank = rank
            assert 0 <= self.rank < self.Ntc

        # 计算 4 种 CFAR 变体的阈值因子
        self.threshold_factor_CA = self.calc_WGN_threshold_factor_CA()
        self.threshold_factor_SOCA = self.calc_WGN_threshold_factor_SOCA()
        self.threshold_factor_GOCA = self.calc_WGN_threshold_factor_GOCA()
        self.threshold_factor_OS = self.calc_WGN_threshold_factor_OS()

        self.params = {
            "CA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_CA),
            "SOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_SOCA),
            "GOCA": (self.Ntc // 2, self.Ngc // 2, self.threshold_factor_GOCA),
            "OS": (self.Ntc // 2, self.Ngc // 2, self.rank, self.threshold_factor_OS),
        }
        self.detector = {
            "CA": cfar.ca,
            "SOCA": cfar.soca,
            "GOCA": cfar.goca,
            "OS": cfar.os,
        }
        self.detector2 = {
            "CA": cfar.ca2,
            "SOCA": cfar.soca2,
            "GOCA": cfar.goca2,
            "OS": cfar.os2,
        }

    def __str__(self):
        return "".join(
            [
                "CFAR 检测器信息\n",
                "=========================\n",
                "训练单元数量: {}\n".format(self.Ntc),
                "保护单元数量: {}\n".format(self.Ngc),
                "虚警概率: {}\n".format(self.Pfa),
                "有序统计秩: {}\n".format(self.rank),
                "阈值因子:\n",
                "      CA-CFAR: {:.3f}\n".format(self.threshold_factor_CA),
                "    SOCA-CFAR: {:.3f}\n".format(self.threshold_factor_SOCA),
                "    GOCA-CFAR: {:.3f}\n".format(self.threshold_factor_GOCA),
                "    OSCA-CFAR: {:.3f}\n".format(self.threshold_factor_OS),
            ]
        )

    def calc_WGN_threshold_factor_CA(self):
        return self.Ntc * (self.Pfa ** (-1.0 / self.Ntc) - 1)

    def calc_WGN_threshold_factor_SOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_SOCA, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("未找到 SOCA 的阈值因子")

    def calc_WGN_threshold_factor_GOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_GOCA, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("未找到 GOCA 的阈值因子")

    def calc_WGN_threshold_factor_OS(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_OS, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("未找到 OS 的阈值因子")

    def calc_WGN_pfa_GOSOCA_core(self, x):
        x = float(x)
        temp = 0.0
        for k in range(int(self.Ntc / 2)):
            l1 = math.lgamma(self.Ntc / 2 + k)
            l2 = math.lgamma(k + 1)
            l3 = math.lgamma(self.Ntc / 2)
            temp += math.exp(l1 - l2 - l3) * (2 + x / (self.Ntc / 2)) ** (-k)
        return temp * (2 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)

    def calc_WGN_pfa_SOCA(self, x):
        return self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_GOCA(self, x):
        x = float(x)
        temp = (1.0 + x / (self.Ntc / 2)) ** (-self.Ntc / 2)
        return temp - self.calc_WGN_pfa_GOSOCA_core(x) - self.Pfa / 2

    def calc_WGN_pfa_OS(self, x):
        l1 = math.lgamma(self.Ntc + 1)
        l2 = math.lgamma(self.Ntc - self.rank + 1)
        l4 = math.lgamma(x + self.Ntc - self.rank + 1)
        l6 = math.lgamma(x + self.Ntc + 1)
        return math.exp(l1 - l2 + l4 - l6) - self.Pfa

    def detect(self, mat, alg="CA"):
        """
        返回目标掩码数组。
        """
        return self.detector[alg](mat, *self.params[alg])

    def detect2(self, mat, alg="CA"):
        """
        返回目标掩码数组和阈值数组。
        """
        return self.detector2[alg](mat, *self.params[alg])