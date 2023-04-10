#!/usr/bin/env python3

import boxx
import torch
from boxx import np


def apply_rectify_on_uv(uvs, K, R):

    """prompt:
    已知一个相机图像上的多个点 uvs (n*[u ,v]), 相机内参 K(3*3), 对相机做一个 R 的旋转, 求旋转后的 uvs_rotate, 用 PyTorch 实现
    """

    uv1 = torch.cat((uvs, torch.ones(uvs.shape[0], 1)), dim=-1)  # n * [u, v, 1]

    # 用相机内参矩阵 K 的逆矩阵将图像坐标转换为相机坐标
    K_inv = torch.inverse(K)
    pts_camera = torch.matmul(uv1, K_inv.t())  # n * [x, y, z]

    # 在相机坐标系中应用旋转矩阵 R
    pts_camera_rotated = torch.matmul(pts_camera, R.t())  # n * [x', y', z']

    # 用相机内参矩阵 K 将旋转后的相机坐标转换回图像坐标
    uv1_rotated = torch.matmul(pts_camera_rotated, K.t())  # n * [u', v', z']

    # 将旋转后的齐次图像坐标转换为非齐次坐标
    uvs_rotated = uv1_rotated[:, :2] / uv1_rotated[:, 2].unsqueeze(-1)
    return uvs_rotated


if __name__ == "__main__":
    from boxx import *

    with boxx.inpkg():
        from . import calibdiff_utils
        from . import rt_to_rectify
    d = calibdiff_utils.get_test_data()

    R, t = d["R"], d["t"]
    re = rt_to_rectify.stereo_recitfy(R, t)
    R1 = re["R1"]

    print(re["R1"])
    print(re["R2"])

    uv_pairs = d["uv_pairs"]
    uvs = uv_pairs[:, :2]
    K = d["K1"]
    R = re["R1"]

    uvs_rotated = apply_rectify_on_uv(uvs, K, R)
