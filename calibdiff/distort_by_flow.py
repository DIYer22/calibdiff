#!/usr/bin/env python3
import boxx
import torch
import calibrating


class DistortByFlow(torch.nn.Module):
    def __init__(self, hw, arg2=.1):
        super().__init__()
        self.hw = hw
        self.arg2 = arg2
        self.param = torch.nn.Parameter(torch.zeros([int(round(i*arg2)) for i in hw]))
        # self.undistort_flow = # resize param to hw using bicubic (h, w)

    def undistort_points(self, uvs):
        """
        # TODO 更科学的插值, 最好基于 image_points 为基础点来插值

        uvs: shape (n, 2)
        return uvs_new(n, 2)
        if uv in hw, return uv + self.undistort_flow[v, u]
        else returnu uv
        """
        self.undistort_flow = F.interpolate(self.param.unsqueeze(0).unsqueeze(0), size=hw, mode='bicubic', align_corners=False).squeeze(0).squeeze(0)

        mask = (0 <= u) & (u < self.hw[1]) & (0 <= v) & (v < self.hw[0])
        assert mask.all()
        uvs_ = uvs.round().long()
        return uvs + self.undistort_flow[uvs_[0],uvs_[1]]

    def distort(self, img):
        self
    
    def undistort(self, img):
        self


if __name__ == "__main__":
    from boxx.ylth import *
    
    
    example_type = "aruco"
    caml, camr, camd = calibrating.get_test_cams(example_type).values()
    
    
    cam = caml
    uv_ds = np.concatenate(cam.image_points, 0)
    
    
    distort = DistortByFlow(hw=cam.xy[::-1])
    distort.undistort_points(uvs_ds)
    
