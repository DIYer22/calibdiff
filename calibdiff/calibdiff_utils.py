#!/usr/bin/env python3

import boxx
from boxx import np
import torch
from torch.autograd import Variable
import cv2

with boxx.inpkg():
    from .apply_rectify_on_uv import apply_rectify_on_uv

eps = 1e-8


def rodrigues_pytorch(rvec):
    theta = torch.norm(rvec) + eps
    # if theta == 0:
    #     return torch.eye(3).type_as(rvec)

    k = rvec / theta
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    R = (
        torch.eye(3)
        + torch.sin(theta) * K
        + (1 - torch.cos(theta)) * torch.matmul(K, K)
    )
    return R


class DifferentiableRotateByRodrigues:
    def to_r(R):
        return cv2.Rodrigues(boxx.npa(R))[0]

    def to_R(r):
        return rodrigues_pytorch(r)


def continuity_rotation_6d(ma_n_1):
    """

    Parameters
    ----------
    ma_n_1 : n * (n-1) array
        3x2: Continuity 6d Rotation Representations.

    Returns
    -------
    ma_nn : n * n
        3*3 rotation array
    """
    h, w = ma_n_1.shape
    ma_nn_cols = list(torch.zeros(size=(h, h)))
    N = lambda v: v / (v**2).sum() ** 0.5
    for i in range(w):
        if i:
            ai = ma_n_1[:, i]
            v = ai[:]
            for j in range(i):
                bj = ma_nn_cols[j]
                v = v - (bj @ ai) * bj
        else:
            v = ma_n_1[:, i]
        ma_nn_cols[i] = N(v)
    cross = ma_nn_cols[0]
    for i in range(1, w):
        cross = torch.cross(cross, ma_nn_cols[i])
    ma_nn_cols[-1] = cross
    ma_nn = torch.stack(ma_nn_cols).T
    # ma_nn.mean().backward()/0
    return ma_nn


class DifferentiableRotateByContinuityRotation:
    def to_r(R):
        return R[:3, :2]

    def to_R(r):
        return continuity_rotation_6d(r)


DifferentiableRotate = DifferentiableRotateByContinuityRotation
# DifferentiableRotate = DifferentiableRotateByRodrigues

def generate_K(fx, cx, fy, cy):
    K = torch.stack(
        [
            torch.cat([fx.unsqueeze(0), torch.tensor([0.0]), cx.unsqueeze(0)]),
            torch.cat([torch.tensor([0.0]), fy.unsqueeze(0), cy.unsqueeze(0)]),
            torch.tensor([0.0, 0.0, 1.0]),
        ],
        dim=0,
    )
    return K


def get_test_data():
    # 生成合成的内参
    fx1, fy1, cx1, cy1, h1, w1 = 800, 800, 640, 480, 960, 1280
    fx2, fy2, cx2, cy2, h2, w2 = 800, 800, 640, 480, 960, 1280

    # 生成合成的外参
    R = cv2.Rodrigues(np.array([0.03, 0.05, 0.04]))[0]
    # R = cv2.Rodrigues(np.array([0.0, 0.0, 0.0]))[0]
    r = DifferentiableRotate.to_r(R)
    t = np.array([-0.2, 0.0, -0.0])  # 平移向量

    # 生成合成的 points
    num_points = 50
    points = np.zeros((num_points, 4))
    for i in range(num_points):
        # 随机生成三维点（单位：米）
        point_3d = np.random.rand(3) * 5

        # 将三维点投影到左右相机的像素平面
        point_left = np.array([fx1, fy1]) * point_3d[:2] / point_3d[2] + np.array(
            [cx1, cy1]
        )
        point_right = np.array([fx2, fy2]) * (point_3d[:2] + t[:2]) / (
            point_3d[2] + t[2]
        ) + np.array([cx2, cy2])

        # 存储在 points 中
        points[i] = [point_left[1], point_left[0], point_right[1], point_right[0]]

    # 将生成的合成数据转换为 PyTorch 张量
    fx1, fy1, cx1, cy1, h1, w1 = map(torch.tensor, [fx1, fy1, cx1, cy1, h1, w1])
    fx2, fy2, cx2, cy2, h2, w2 = map(torch.tensor, [fx2, fy2, cx2, cy2, h2, w2])
    r, t = map(torch.tensor, [r, t])
    uv_pairs = torch.tensor(points, dtype=torch.float)

    # 将参数设为可优化的变量
    d = dict(uv_pairs=uv_pairs)
    d["param"] = (
        d["fx1"],
        d["fy1"],
        d["cx1"],
        d["cy1"],
        d["h1"],
        d["w1"],
        d["fx2"],
        d["fy2"],
        d["cx2"],
        d["cy2"],
        d["h2"],
        d["w2"],
        d["r"],
        d["t"],
    ) = [
        Variable(p.float(), requires_grad=True)
        for p in [fx1, fy1, cx1, cy1, h1, w1, fx2, fy2, cx2, cy2, h2, w2, r, t]
    ]

    d["R"] = DifferentiableRotate.to_R(d["r"])
    # for i in range(1,3):
    #     d[f'K{i}'] = torch.tensor([[d[f'fx{i}'], 0, d[f'cx{i}']],[0, d[f'fy{i}'], d[f'cy{i}']],[0,0,1]], requires_grad=True)

    for i in range(1, 3):
        d[f"K{i}"] = generate_K(d[f"fx{i}"], d[f"cx{i}"], d[f"fy{i}"], d[f"cy{i}"])
    return d


if __name__ == "__main__":
    from boxx import *
