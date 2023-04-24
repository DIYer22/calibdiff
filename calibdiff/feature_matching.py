#!/usr/bin/env python3

import cv2
import boxx
import torch
import numpy as np


with boxx.inpkg():
    from .calibdiff_utils import vis_matched_uvs, try_load_img


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
        img1, img2 = try_load_img(img1), try_load_img(img2)
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
        print(f"{type(self).__name__}.process_img_path_pairs():")
        uv1s, uv2s, confidences = [], [], []
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


def mask_to_udlr(mask):
    """
    >>> u,d,l,r = mask_to_udlr(mask)
    """
    try:
        import pycocotools.mask

        rle = pycocotools.mask.encode(np.asfortranarray(np.uint8(mask)))
        bbox = pycocotools.mask.toBbox(rle).round().astype(int).tolist()
        u, d, l, r = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
        return u, d, l, r
    except:
        pass
    # 7x slow
    h, w = mask.shape
    y, x = np.mgrid[0:h, 0:w]
    if mask.dtype != np.bool:
        mask = mask > 0
    ys = y[mask]
    xs = x[mask]
    u, d = ys.min(), ys.max() + 1
    l, r = xs.min(), xs.max() + 1
    return u, d, l, r


def expand_bbox(udlr, hw, rate=0.2):
    u, d, l, r = udlr
    h, w = hw
    # 计算bbox的宽度和高度
    bbox_width = r - l
    bbox_height = d - u

    # 计算膨胀大小
    expansion_w = int(bbox_width * rate)
    expansion_h = int(bbox_height * rate)

    # 扩展bbox
    u = max(0, u - expansion_h)
    d = min(h, d + expansion_h)
    l = max(0, l - expansion_w)
    r = min(w, r + expansion_w)

    # 返回扩展后的bbox
    return u, d, l, r


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
        # resize_shape = (480, 640)
        resize_shape = (768, 1024)
        # resize_shape = (576, 768)
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

    def __call__(self, img1, img2, mask1=None, mask2=None):
        if mask1 is not None:
            u, d, l, r = udlr1 = expand_bbox(mask_to_udlr(mask1), mask1.shape)
            img1, img_raw1 = img1[u:d, l:r], img1
        if mask2 is not None:
            u, d, l, r = udlr2 = expand_bbox(mask_to_udlr(mask2), mask2.shape)
            img2, img_raw2 = img2[u:d, l:r], img2

        to_th_img = (
            lambda img: self.kornia.image_to_tensor(
                np.ascontiguousarray(img[..., ::-1]), False
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
            uvs1 * img1.shape[:2][::-1],
            uvs2 * img2.shape[:2][::-1],
            cv2.USAC_MAGSAC,
            1.0,
            0.99995,
            100000,
        )
        # TODO why 0.99 => .75 works for aruco?
        idxs = inliers[:, 0] > -10
        topk = self.cfg.get("topk", len(uvs1))
        if topk <= 1:
            topk = int(idxs.sum() * topk + 1)

        # 找到第topk个True值的索引
        true_indices = np.where(idxs)[0]
        if topk < len(true_indices):
            idx_to_change = true_indices[topk - 1]  # 索引从0开始，因此要减1
            # 将第topk个True值及其之后的所有True值都变为False
            idxs[idx_to_change + 1 :] = False

        if mask1 is not None:
            u, d, l, r = udlr1
            h, w = img_raw1.shape[:2]
            uvs1 = uvs1 * [[(r - l) / w, (d - u) / h]] + [[l / w, u / h]]
            img1 = img_raw1
        if mask2 is not None:
            u, d, l, r = udlr2
            h, w = img_raw2.shape[:2]
            uvs2 = uvs2 * [[(r - l) / w, (d - u) / h]] + [[l / w, u / h]]
            img2 = img_raw2

        matched = dict(
            uvs1=uvs1[idxs],
            uvs2=uvs2[idxs],
            confidence=confidence[idxs],
            img1=img1,
            img2=img2,
        )
        boxx.mg()
        return matched


if __name__ == "__main__":
    import calibrating
    from boxx.ylth import *

    caml, camr, camd = calibrating.get_test_cams("aruco").values()

    key = list(caml)[0]
    img1 = boxx.imread(caml[key]["path"])
    img2 = boxx.imread(camr[key]["path"])
    # img1 = boxx.imread("/home/dl/ai_asrs/2112_huafeng/big_file/l.jpg")
    # img2 = boxx.imread("/home/dl/ai_asrs/2112_huafeng/big_file/r.jpg")
    # img2 = np.uint8(
    #     __import__("skimage.transform").transform.rotate(img1, angle=30) * 255.5
    # )
    # img2 = np.rot90(img2, k=1)
    feature_matching = LoftrFeatureMatching(dict(topk=0.999))
    mask1, mask2 = None, None
    if "mask" and 0:
        mask1 = caml.get_calibration_board_depth(img1)["depth"] > 0
        mask2 = camr.get_calibration_board_depth(img2)["depth"] > 0
    matched = feature_matching(img1, img2, mask1, mask2)
    uvs1, uvs2, confidence = (
        matched["uvs1"],
        matched["uvs2"],
        matched["confidence"],
    )

    vis_matched = feature_matching.vis(matched)
    boxx.showb - vis_matched
