#!/usr/bin/env python3

import cv2
import boxx
import torch
import numpy as np

with boxx.inpkg():
    from .calibdiff_utils import vis_matched_uvs


class MetaFeatureMatching:
    def __init__(self, cfg=None):
        self.cfg = cfg
        # self.matcher = build

    def __call__(self, img1, img2):
        # input: RGB uint8 (h, w, 3)uint8
        raise NotImplementedError()
        # output: float disparity (h, w)float64, unit is m
        # return uv_pairs

    def vis(self, matched=None, img1=None, img2=None):
        if matched is None:
            matched = self(img1, img2)
        uvs1, uvs2, confidence = (
            matched["uvs1"],
            matched["uvs2"],
            matched.get("confidence"),
        )
        img1 = matched.get("img1", img1)
        img2 = matched.get("img2", img2)
        vis_matched = vis_matched_uvs(uvs1, uvs2, img1, img2, confidence)
        return vis_matched

    def process_img_path_pairs(self, img_path_pairs):
        uv1s, uv2s, confidences = [], [], []
        print(f"{type(self).__name__}.process_img_path_pairs():")
        for imgp1, imgp2 in __import__("tqdm").tqdm(list(img_path_pairs)):
            img1 = boxx.imread(imgp1)
            img2 = boxx.imread(imgp2)
            matched = self(img1, img2)
            uv1s.append(matched["uvs1"] * img1.shape[:2][::-1])
            uv2s.append(matched["uvs2"] * img2.shape[:2][::-1])
            if "confidence" in matched:
                confidences.append(matched["confidence"])
        uvs1, uvs2 = np.concatenate(uv1s, 0), np.concatenate(uv2s, 0)
        matched = dict(uvs1=uvs1, uvs2=uvs2)
        if confidences:
            matched["confidence"] = np.concatenate(confidences, 0)
        return matched


def get_uv_pairs_from_stereo_boards(stereo):
    uv1s, uv2s, objps = stereo.get_conjoint_points()
    uvs1, uvs2 = np.concatenate(uv1s, 0), np.concatenate(uv2s, 0)
    return uvs1, uvs2


class LoftrFeatureMatching(MetaFeatureMatching):
    def __init__(self, cfg=None):
        import kornia

        self.kornia = kornia
        self.cfg = cfg or {}
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.matcher = kornia.feature.LoFTR(pretrained="indoor_new").to(self.device)

    def matching_in_torch(self, th_img1, th_img2):
        resize_shape = (480, 640)
        # if boxx.mg():
        #     resize_shape = (240, 320)

        img1 = self.kornia.geometry.resize(th_img1, resize_shape, antialias=True)
        img2 = self.kornia.geometry.resize(th_img2, resize_shape, antialias=True)
        input_dict = {
            "image0": self.kornia.color.rgb_to_grayscale(img1).to(
                self.device
            ),  # LofTR works on grayscale images only
            "image1": self.kornia.color.rgb_to_grayscale(img2).to(self.device),
        }
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

            shape_xy = torch.tensor(resize_shape[::-1]).float().to(self.device)
            correspondences["confidence"], correspondences["sorted_idx"] = torch.sort(
                correspondences["confidence"], descending=True
            )
            correspondences["keypoints0"] /= shape_xy
            correspondences["keypoints1"] /= shape_xy
            correspondences["keypoints0"] = correspondences["keypoints0"][
                correspondences["sorted_idx"]
            ]
            correspondences["keypoints1"] = correspondences["keypoints1"][
                correspondences["sorted_idx"]
            ]

        return correspondences

    def __call__(self, img1, img2):
        to_th_img = (
            lambda img: self.kornia.image_to_tensor(
                np.ascontiguousarray(img), False
            ).float()
            / 255.0
        )
        th_img1 = to_th_img(img1)
        th_img2 = to_th_img(img2)
        correspondences = self.matching_in_torch(th_img1, th_img2)

        uvs1 = correspondences["keypoints0"].cpu().numpy()
        uvs2 = correspondences["keypoints1"].cpu().numpy()
        confidence = correspondences["confidence"].cpu().numpy()
        assert uvs1.size > 20, f"Too few matched points {uvs1}"
        Fm, inliers = cv2.findFundamentalMat(
            uvs1 * img1.shape[:2],
            uvs2 * img2.shape[:2],
            cv2.USAC_MAGSAC,
            1.0,
            0.999,
            100000,
        )
        idxs = inliers[:, 0] > 0
        topk = self.cfg.get("topk", len(uvs1))
        if topk <= 1:
            topk = int(idxs.sum() * topk + 1)

        # 找到第topk个True值的索引
        true_indices = np.where(idxs)[0]
        if topk < len(true_indices):
            idx_to_change = true_indices[topk - 1]  # 索引从0开始，因此要减1
            # 将第topk个True值及其之后的所有True值都变为False
            idxs[idx_to_change + 1 :] = False
        boxx.mg()
        matched = dict(
            uvs1=uvs1[idxs],
            uvs2=uvs2[idxs],
            confidence=confidence[idxs],
            img1=img1,
            img2=img2,
        )
        return matched


if __name__ == "__main__":
    import calibrating
    from boxx.ylth import *

    caml, camr, camd = calibrating.get_test_cams("aruco").values()

    key = list(caml)[0]
    img1 = boxx.imread(caml[key]["path"])
    img2 = boxx.imread(camr[key]["path"])
    img2 = np.uint8(
        __import__("skimage.transform").transform.rotate(img1, angle=30) * 255.5
    )
    # img2 = np.rot90(img2, k=1)
    feature_matching = LoftrFeatureMatching(dict(topk=100))
    matched = feature_matching(img1, img2)
    uvs1, uvs2, confidence = (
        matched["uvs1"],
        matched["uvs2"],
        matched["confidence"],
    )

    vis_matched = feature_matching.vis(matched)
    boxx.show - vis_matched
