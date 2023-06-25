#!/usr/bin/env python3
import boxx
import torch
import numpy as np
from boxx import strnum, npa
from boxx.ylth import tht
from calibrating import R_t_to_T, T_to_deg_distance


class LidarToCam:
    def __init__(
        self, T_board2cams, xyzs_board_in_lidars, T_lidar2cam_init=None, cfg=None
    ):
        self.cfg = cfg or {}
        if T_lidar2cam_init is None:
            T_lidar2cam_init = np.eye(4)
        self.T_lidar2cam_init = T_lidar2cam_init.copy()
        self.T_board2cams = T_board2cams
        self.xyzs_board_in_lidars = xyzs_board_in_lidars
        assert len(T_board2cams) == len(xyzs_board_in_lidars)

    def optimize(self):
        import calibdiff

        R_init = self.T_lidar2cam_init[:3, :3].copy()
        t_init = self.T_lidar2cam_init[:3, 3].copy()

        r = calibdiff.DifferentiableRotateByContinuityRotation.to_r(R_init)
        T_cam2boards = [tht(np.linalg.inv(T)) for T in self.T_board2cams]
        xyz_lidars = [tht(xyz_lidar) for xyz_lidar in self.xyzs_board_in_lidars]
        r = tht(r).requires_grad_()
        t = tht(t_init).requires_grad_()
        param = dict(r=r, t=t)
        optimizer = torch.optim.Adam(param.values(), lr=1e-3)
        # optimizer = torch.optim.SGD(param.values(), momentum=0.9, lr=1e-3)
        num_iterations = 1000
        mean_zs = lambda: torch.abs(zs_).mean().item()
        for idx in range(num_iterations):
            optimizer.zero_grad()
            R = calibdiff.DifferentiableRotateByContinuityRotation.to_R(r)
            zs = []
            for xyz_lidar, T_cam2board in zip(xyz_lidars, T_cam2boards):
                xyz_in_cam = xyz_lidar @ R.T + t[None]
                xyz_in_board = xyz_in_cam @ T_cam2board[:3, :3].T + T_cam2board[:3, 3]
                z = xyz_in_board[:, 2]
                zs.append(z)

            zs_ = torch.cat(zs, 0)
            loss = (zs_**2).mean()
            loss.backward()
            if idx % 100 == 0:  # boxx.mg():
                # print({k: v.grad for k, v in param.items()})
                print(
                    f"Iteration {idx}: mean_z={strnum(mean_zs())}"  # , {', '.join([f'{k}={strnum(lossd[k])}' for k in lossd])}"
                )
            optimizer.step()
        T_res = R_t_to_T(npa(R), npa(t))
        T_delta = T_to_deg_distance(T_res, self.T_lidar2cam_init)
        return dict(
            T=T_res,
            mean_distance=mean_zs(),
            boardn=len(T_cam2boards),
            T_delta={k: T_delta[k] for k in T_delta if k in ["deg", "distance"]},
        )

    __call__ = optimize


if __name__ == "__main__":
    from calibrating import Cam, Stereo, glob, apply_T_to_point_cloud
    from calibrating import (
        convert_points_for_cv2,
        prepare_example_data_dir,
        PredifinedArucoBoard1,
    )
    from boxx import *
    import os
    from boxx import imread

    example_data_dir = os.path.join(
        prepare_example_data_dir(), "paired_stereo_and_depth_cams_aruco"
    )
    board = PredifinedArucoBoard1()
    caml = Cam(
        glob(os.path.join(example_data_dir, "*", "stereo_l.jpg")),
        board,
        name="caml",
        enable_cache=True,
    )
    camd = Cam(
        glob(os.path.join(example_data_dir, "*", "depth_cam_color.jpg")),
        board,
        name="camd",
        enable_cache=True,
    )
    camd_built_in_intrinsics = dict(
        fx=1474.1182177692722,
        fy=1474.125874583105,
        cx=1037.599716850734,
        cy=758.3072639103259,
    )
    # depth need to be used in pairs with camera's built-in intrinsics
    camd.load(camd_built_in_intrinsics)

    cam = caml

    key = sorted(camd)[0]
    dicd = camd[key]
    imgpd = dicd["path"]
    imgd = imread(imgpd)
    depthd_board = camd.get_calibration_board_depth(imgd)["depth"]
    depthd = imread(imgpd.replace("color.jpg", "depth.png")) / 1000.0
    # shows(imgd, vis_depth_l1(depthd, depthd_board, 0.005))

    stereo = Stereo(cam, camd)
    board = cam.board
    # generate test data
    T_lidar2cam_init = np.eye(4)
    T_cam_in_lidar = R_t_to_T(stereo.R, stereo.t)
    T_gt = np.linalg.inv(T_cam_in_lidar)

    T_board2cams = [d["T"] for d in cam.values() if "T" in d]
    xyzs_board_in_lidars = [
        apply_T_to_point_cloud(
            T_cam_in_lidar @ d["T"], convert_points_for_cv2(d["object_points"])
        )
        for d in cam.values()
        if "T" in d
    ]
    res = LidarToCam(T_board2cams, xyzs_board_in_lidars, T_lidar2cam_init)()
    print(res)
    print(T_gt)
