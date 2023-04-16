#!/usr/bin/env python3

import boxx
import torch
import torch.optim as optim
from boxx import npa, np
import calibrating

with boxx.inpkg():
    from . import calibdiff_utils
    from . import rt_to_rectify

device = calibdiff_utils.device


class StereoOptimize:
    def __init__(self, stereo, uvs1, uvs2):
        self.stereo = stereo
        self.uv_pairs = torch.from_numpy(np.concatenate([uvs1, uvs2], 1)).to(device)
        self.context = self.stereo_to_context()

    def stereo_to_context(self):
        context = {}
        cam1 = self.stereo.cam1
        cam2 = self.stereo.cam2
        r = calibdiff_utils.DifferentiableRotate.to_r(self.stereo.R)
        para_ = dict(
            r=r,
            t=self.stereo.t,
            fx1=cam1.fx,
            fy1=cam1.fy,
            cx1=cam1.cx,
            cy1=cam1.cy,
            fx2=cam2.fx,
            fy2=cam2.fy,
            cx2=cam2.cx,
            cy2=cam2.cy,
        )
        param = {
            k: torch.tensor(v).to(device).requires_grad_() for k, v in para_.items()
        }

        context["param"] = param
        return context

    def optimize(self):
        d = self.context
        param = d["param"]
        self.uv_pairs = self.uv_pairs.type_as(param["r"])
        # 设置优化器
        optimizer = optim.Adam(param.values(), lr=1e-3)
        # optimizer = optim.SGD(param.values(), lr=1e-8)

        # 迭代优化
        num_iterations = 1000
        for idx in range(num_iterations):
            optimizer.zero_grad()

            for i in range(1, 3):
                d[f"K{i}"] = calibdiff_utils.generate_K(
                    param[f"fx{i}"], param[f"cx{i}"], param[f"fy{i}"], param[f"cy{i}"]
                )
            R = calibdiff_utils.DifferentiableRotate.to_R(param["r"])
            re = rt_to_rectify.stereo_recitfy(R, param["t"])
            uv1_rectifys = calibdiff_utils.apply_rectify_on_uv(
                self.uv_pairs[:, :2], d["K1"], re["R1"]
            )
            uv2_rectifys = calibdiff_utils.apply_rectify_on_uv(
                self.uv_pairs[:, 2:], d["K2"], re["R2"]
            )
            loss_polar_alignment = torch.abs(
                uv1_rectifys[:, 1] - uv2_rectifys[:, 1]
            ).mean()
            loss = loss_polar_alignment / self.stereo.cam1.xy[1]
            # loss = loss**2
            loss.backward()
            if idx % 100 == 0 and boxx.mg():
                # print({k: v.grad for k, v in param.items()})
                print(
                    f"Iteration {idx}: loss_polar_alignment = {loss_polar_alignment.item()}"
                )
            optimizer.step()

        return calibrating.Stereo.load(
            dict(
                retval=loss_polar_alignment,
                R=npa - R,
                t=npa - param["t"],
                cam1=dict(
                    fx=param["fx1"],
                    cx=param["cx1"],
                    fy=param["fy1"],
                    cy=param["cy1"],
                    xy=self.stereo.cam1.xy,
                ),
                cam2=dict(
                    fx=param["fx2"],
                    cx=param["cx2"],
                    fy=param["fy2"],
                    cy=param["cy2"],
                    xy=self.stereo.cam2.xy,
                ),
            )
        )

    __call__ = optimize


def get_jx_stereo():
    root = "/home/dl/ai_asrs/2011_jingxin_info/big_file_jingxin/calibrating_data/2201.checkboard_img2"
    feature_lib = calibrating.CheckboardFeatureLib((8, 5), 26.03)
    undistorted = False
    caml = calibrating.Cam(
        glob(os.path.join(root, "*/0_color.jpg")),
        feature_lib,
        name="jx-caml",
        enable_cache=True,
        undistorted=undistorted,
        save_feature_vis=False,
    )
    camr = calibrating.Cam(
        glob(os.path.join(root, "*/0_stereo.jpg")),
        feature_lib,
        name="jx-camr",
        enable_cache=True,
        undistorted=undistorted,
        save_feature_vis=False,
    )

    stereo = calibrating.Stereo.load(
        """
R: [[1,0,0.],[0,1,0,],[0.,0,1.]]
_calibrating_version: 0.6.2
cam1:
  cx: 2012
  cy: 1518
  fx: 3230
  fy: 3230
  xy:
  - 4024
  - 3036
cam2:
  cx: 2012
  cy: 1518
  fx: 3230
  fy: 3230
  xy:
  - 4024
  - 3036
t:
- - -0.43
- - 0.0
- - 0.0
"""
    )
    # stereo.R = cv2.Rodrigues(boxx.npa-[0.17]*3)[0]
    return caml, camr, stereo


if __name__ == "__main__":
    from boxx import glob, os

    with boxx.inpkg():
        from .feature_matching import (
            LoftrFeatureMatching,
            get_uv_pairs_from_stereo_boards,
        )
        from boxx import *

    caml, camr, camd = calibrating.get_test_cams().values()

    stereo = calibrating.Stereo.load(
        """
R: [[1,0,0.],[0,1,0,],[0.,0,1.]]
_calibrating_version: 0.6.2
cam1:
  cx: 1368
  cy: 912
  fx: 1316
  fy: 1316
  xy:
  - 2736
  - 1824
cam2:
  cx: 1368
  cy: 912
  fx: 1316
  fy: 1316
  xy:
  - 2736
  - 1824
t:
- - -0.3
- - -0.0
- - 0.0
"""
    )

    # caml, camr, stereo = get_jx_stereo()

    stereo_gt = calibrating.Stereo(caml, camr)
    # stereo = stereo_gt.copy()
    # TODO 为什么在不设置畸变的情况下, 从 gt 继承初始值效果差好多?

    mode = "LoFTR_feature"
    # mode = "boards_feature"
    if mode == "LoFTR_feature":
        feature_matching = LoftrFeatureMatching(dict(topk=0.5))
        img_path_pairs = [
            (
                caml[k]["path"],
                camr[k]["path"],
            )
            for k in caml
            if k in camr
        ]
        matched = feature_matching.process_img_path_pairs(img_path_pairs[:3])
        uvs1, uvs2 = matched["uvs1"], matched["uvs2"]
    elif mode == "boards_feature":
        uvs1, uvs2 = get_uv_pairs_from_stereo_boards(stereo_gt)

    stereo_optimize = StereoOptimize(stereo, uvs1, uvs2)

    stereo_re = stereo_optimize.optimize()
    print("stereo_gt:")
    print(stereo_gt)
    print("stereo_re:")
    print(stereo_re)
    caml.vis_stereo(camr, stereo_gt)
    caml.vis_stereo(camr, stereo_re)
    caml.vis_stereo(camr, stereo)
    # showb - feature_matching.vis(matched)
