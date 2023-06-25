#!/usr/bin/env python3

import boxx
from boxx import np
import torch
from torch.autograd import Variable
import cv2

with boxx.inpkg():
    from .apply_rectify_on_uv import apply_rectify_on_uv

eps = 1e-8
device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")


def torch_tensor(arr, **argkws):
    argkws.setdefault("device", device)
    return torch.tensor(arr, **argkws)


def rodrigues_pytorch(rvec):
    theta = torch.norm(rvec) + eps
    # if theta == 0:
    #     return torch.eye(3).type_as(rvec)

    k = rvec / theta
    K = torch.stack(
        [
            torch_tensor(0.0),
            -k[2],
            k[1],
            k[2],
            torch_tensor(0.0),
            -k[0],
            -k[1],
            k[0],
            torch_tensor(0.0),
        ]
    ).view(3, 3)
    # K = torch_tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    R = (
        torch.eye(3).to(device)
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


def R_t_to_T_torch(R, t=None):
    if t is None:
        t = torch.zeros((3,), device=R.device, dtype=R.dtype)
    t = t.view(3, 1)
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device, dtype=R.dtype).view(
        1, 4
    )
    Rt = torch.cat([R, t], dim=1)
    T = torch.cat([Rt, last_row], dim=0)
    return T


def generate_K(fx, cx, fy, cy):
    K = torch.stack(
        [
            torch.cat([fx.unsqueeze(0), torch_tensor([0.0]), cx.unsqueeze(0)]),
            torch.cat([torch_tensor([0.0]), fy.unsqueeze(0), cy.unsqueeze(0)]),
            torch_tensor([0.0, 0.0, 1.0]),
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
    fx1, fy1, cx1, cy1, h1, w1 = map(torch_tensor, [fx1, fy1, cx1, cy1, h1, w1])
    fx2, fy2, cx2, cy2, h2, w2 = map(torch_tensor, [fx2, fy2, cx2, cy2, h2, w2])
    r, t = map(torch_tensor, [r, t])
    uv_pairs = torch_tensor(points, dtype=torch.float)

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
    #     d[f'K{i}'] = torch_tensor([[d[f'fx{i}'], 0, d[f'cx{i}']],[0, d[f'fy{i}'], d[f'cy{i}']],[0,0,1]], requires_grad=True)

    for i in range(1, 3):
        d[f"K{i}"] = generate_K(d[f"fx{i}"], d[f"cx{i}"], d[f"fy{i}"], d[f"cy{i}"])
    return d


def set_rotate_imread_for_test(rotate_substr="stereo_r.jpg", angle=30):
    def _imread(fname, *l, **kv):
        import skimage.io

        img = skimage.io.imread(fname, *l, **kv)
        if rotate_substr in fname:
            print(fname)
            img = np.uint8(
                __import__("skimage.transform").transform.rotate(img, angle=angle)
                * 255.5
            )
        return img

    boxx.imread = _imread


def try_load_img(arr_or_path):
    if isinstance(arr_or_path, str):
        return boxx.imread(arr_or_path)
    return arr_or_path


"""
pormpt: Write a function to vis points matched in two images
- uvs1 are normalizd xy(0~1) of points in img1
- img are RGB format(h,w,3). if img is None, just replace by a 512*512 black img
- line color between uv1 and uv2 are random Highly saturated colors.
- if has confidence shape(n,), less confidence more tranparent
- support img1 and img2 have different shape
- line thickness are adaptive to img size 
"""


def vis_matched_uvs(uvs1, uvs2, img1=None, img2=None, confidence=None):

    if uvs1.size and uvs1.max() > 1 and uvs2.max() > 1:
        maxx = max(uvs1.max(), uvs2.max())
        uvs1 = uvs1 / ([maxx, maxx] if img1 is None else img1.shape[:2][::-1])
        uvs2 = uvs2 / ([maxx, maxx] if img2 is None else img2.shape[:2][::-1])
    if img1 is None:
        img1 = np.zeros((1024, 1024, 3), dtype=np.uint8) + 192
    if img2 is None:
        img2 = np.zeros((1024, 1024, 3), dtype=np.uint8) + 64
    img1, img2 = try_load_img(img1), try_load_img(img2)
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    # Resize images to have the same height
    if h1 != h2:
        new_w1 = int(w1 * h2 / h1)
        img1 = cv2.resize(img1, (new_w1, h2))
        h1, w1, _ = img1.shape

    # Create a black canvas with the same height as img1 and img2, and the sum of their widths
    canvas = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
    canvas[:, :w1, :] = img1
    canvas[:, w1:, :] = img2
    if not uvs1.size:
        return canvas

    if confidence is not None:
        if confidence.ndim == 1:
            confidence = np.expand_dims(confidence, axis=1)
        assert (
            uvs1.shape[0] == confidence.shape[0]
        ), "Confidence values must match the number of points in uvs1"
    # First, draw all lines
    line_colors = boxx.getDefaultColorList(12, uint8=True) * (len(uvs1) // 10 + 1)
    for i, (uv1, uv2) in enumerate(zip(uvs1, uvs2)):
        x1, y1 = int(uv1[0] * w1), int(uv1[1] * h1)
        x2, y2 = int(uv2[0] * w2) + w1, int(uv2[1] * h2)

        line_color = line_colors[i]
        line_thickness = max(1, int(min(w1, w2) * 0.001))

        if confidence is not None:
            alpha = max(0.2, min(1, confidence[i]))
            line_color = tuple(int(c * alpha) for c in line_color)

        cv2.line(canvas, (x1, y1), (x2, y2), line_color, thickness=line_thickness)

    # Then, draw all points
    for i, (uv1, uv2) in enumerate(zip(uvs1, uvs2)):
        x1, y1 = int(uv1[0] * w1), int(uv1[1] * h1)
        x2, y2 = int(uv2[0] * w2) + w1, int(uv2[1] * h2)

        line_color = line_colors[i]
        point_radius = max(1, int(min(w1, w2) * 0.0015))

        if confidence is not None:
            alpha = max(0.2, min(1, confidence[i]))
            line_color = tuple(int(c * alpha) for c in line_color)

        cv2.circle(canvas, (x1, y1), point_radius * 2, (255, 255, 255), thickness=-1)
        cv2.circle(canvas, (x2, y2), point_radius * 2, (255, 255, 255), thickness=-1)
        cv2.circle(canvas, (x1, y1), point_radius, line_color, thickness=-1)
        cv2.circle(canvas, (x2, y2), point_radius, line_color, thickness=-1)

    return canvas


if __name__ == "__main__":
    from boxx import *
