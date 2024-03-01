#!/usr/bin/env python3
import cv2
import boxx
import torch
import calibrating
import numpy as np
import torch.nn.functional as F


def flow_to_remap(flow):
    _, h, w = flow.shape
    # remap_flow = flow.transpose(2, 0, 1)
    remap_flow = flow * [[[w]], [[h]]]
    remap_xy = np.float32(np.mgrid[:h, :w][::-1])
    uv_new = (remap_xy + remap_flow).round().astype(np.int32)
    mask = (uv_new[0] >= 0) & (uv_new[1] >= 0) & (uv_new[0] < w) & (uv_new[1] < h)
    uv_new_ = uv_new[:, mask]
    remap_xy[:, uv_new_[1], uv_new_[0]] = remap_xy[:, mask]
    mask_remaped = np.zeros((h, w), np.bool8)
    mask_remaped[uv_new_[1], uv_new_[0]] = True
    return remap_xy, mask_remaped


class DistortByFlow(torch.nn.Module):
    def __init__(self, hw, arg2=0.1):
        super().__init__()
        self.hw = hw
        self.arg2 = arg2
        self.param = torch.nn.Parameter(
            torch.zeros([2] + [int(round(i * arg2)) for i in hw])
        )
        # shape 为 (2, h, w), 值域为 [-1, +1] 的 float numpy
        # self.undistort_flow = # resize param to hw using bicubic (h, w)

    def get_undistort_flow(self):
        return F.interpolate(
            self.param[None], size=self.hw, mode="bicubic", align_corners=False
        )[0]

    def undistort_points(self, uvs, undistort_flow=None):
        """
        # TODO 更科学的插值, 最好基于 image_points 为基础点来插值

        uvs: shape (n, 2)
        return uvs_new(n, 2)
        if uv in hw, return uv + self.undistort_flow[v, u]
        else returnu uv
        """
        if isinstance(uvs, np.ndarray):
            uvs = torch.from_numpy(uvs)

        self.undistort_flow = (
            self.get_undistort_flow() if undistort_flow is None else undistort_flow
        )

        uvs_ = uvs.round().long()
        u, v = (
            uvs_[:, 0],
            uvs_[:, 1],
        )
        mask = (0 <= u) & (u < self.hw[1]) & (0 <= v) & (v < self.hw[0])
        assert mask.all()
        return (
            uvs
            + (
                self.undistort_flow[:, v, u]
                * torch.tensor([self.hw[1], self.hw[0]])[:, None].float()
            ).T
        )

    def distort_img(self, img):
        self

    def undistort_img(self, img, interpolation=cv2.INTER_LINEAR):

        if not hasattr(self, "remap"):
            self.undistort_flow = self.get_undistort_flow()
            self.remap, self.mask_remaped = flow_to_remap(
                boxx.npa(self.undistort_flow),
            )
        img_new = (
            cv2.remap(img, *self.remap, interpolation=interpolation)
            * self.mask_remaped[..., None]
        )

        return img_new


if __name__ == "__main__":
    from boxx.ylth import *

    example_type = "aruco"
    caml, camr, camd = calibrating.get_test_cams(example_type).values()

    cam = caml
    key = list(cam)[0]
    path = cam[key]["path"]
    img = imread(path)
    uv_ds = np.concatenate(cam.image_points, 0)

    distort = DistortByFlow(hw=cam.xy[::-1])
    uv_unds = distort.undistort_points(uv_ds)
    img_undisotrt = distort.undistort_img(img)
