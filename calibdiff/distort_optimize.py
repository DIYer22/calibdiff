#!/usr/bin/env python3
import boxx
from boxx import np
import calibrating
import torch
import torch.optim as optim


def undistort(uv_distorts, k, D, iter_num=3):
    xys = uv_distorts.clone()
    k1, k2, p1, p2, k3 = D[0] if len(D) == 1 else D
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    xys[:, 0] = (xys[:, 0] - cx) / fx
    xys[:, 1] = (xys[:, 1] - cy) / fy
    xys0 = xys.clone()
    for _ in range(iter_num):
        r2 = xys[:, 0] ** 2 + xys[:, 1] ** 2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * xys[:, 0] * xys[:, 1] + p2 * (r2 + 2 * xys[:, 0] ** 2)
        delta_y = p1 * (r2 + 2 * xys[:, 1] ** 2) + 2 * p2 * xys[:, 0] * xys[:, 1]
        xs = (xys0[:, 0] - delta_x) * k_inv
        ys = (xys0[:, 1] - delta_y) * k_inv
        xys = torch.cat([xs[:, None], ys[:, None]], 1)
        # g()/0
    xys[:, 0] = xys[:, 0] * fx + cx
    xys[:, 1] = xys[:, 1] * fy + cy
    return xys


def undistort(uv_distorts, k, D, iter_num=3):
    xys = uv_distorts.clone()
    k1, k2, p1, p2, k3 = D[0] if len(D) == 1 else D
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    xys = torch.stack(((xys[:, 0] - cx) / fx, (xys[:, 1] - cy) / fy), dim=1)
    xys0 = xys.clone()
    for _ in range(iter_num):
        r2 = xys[:, 0] ** 2 + xys[:, 1] ** 2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * xys[:, 0] * xys[:, 1] + p2 * (r2 + 2 * xys[:, 0] ** 2)
        delta_y = p1 * (r2 + 2 * xys[:, 1] ** 2) + 2 * p2 * xys[:, 0] * xys[:, 1]
        xs = (xys0[:, 0] - delta_x) * k_inv
        ys = (xys0[:, 1] - delta_y) * k_inv
        xys = torch.cat([xs[:, None], ys[:, None]], 1)
    xys = torch.stack((xys[:, 0] * fx + cx, xys[:, 1] * fy + cy), dim=1)
    return xys


def undistort_point(xy, k, distortion, iter_num=3):
    k1, k2, p1, p2, k3 = distortion
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    x, y = xy.astype(float)
    x = (x - cx) / fx
    x0 = x
    y = (y - cy) / fy
    y0 = y
    for _ in range(iter_num):
        r2 = x**2 + y**2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        x = (x0 - delta_x) * k_inv
        y = (y0 - delta_y) * k_inv
    return np.array((x * fx + cx, y * fy + cy))


def distort(xy, k, D):
    k1, k2, p1, p2, k3 = D[0] if len(D) == 1 else D
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]
    i, j = xy
    x = (i - cx) / fx
    y = (j - cy) / fy
    r2 = x * x + y * y

    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    scale = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

    xBis = x * scale + dx
    yBis = y * scale + dy

    xCorr = xBis * fx + cx
    yCorr = yBis * fy + cy
    return xCorr, yCorr


def distort(xys, k, D):
    k1, k2, p1, p2, k3 = D[0] if len(D) == 1 else D
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[:2, 2]

    # Extract the x and y coordinates from the xys array
    i = xys[:, 0]
    j = xys[:, 1]

    x = (i - cx) / fx
    y = (j - cy) / fy
    r2 = x * x + y * y

    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    scale = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

    xBis = x * scale + dx
    yBis = y * scale + dy

    xCorr = xBis * fx + cx
    yCorr = yBis * fy + cy

    # Join the corrected x and y coordinates into a 2D array and return it
    return torch.column_stack((xCorr, yCorr))


if __name__ == "__main__":
    from boxx.ylth import *

    example_type = "aruco"
    caml, camr, camd = calibrating.get_test_cams(example_type).values()
    cam = caml
    uv_ds = np.concatenate(cam.image_points, 0)
    uv_unds = cam.undistort_points(uv_ds)

    using_distort = True
    using_distort = False
    if not using_distort:
        param = {"D": tht(cam.D * 0).requires_grad_()}
        uv_ds = tht(uv_ds.copy())
        uv_unds = tht(uv_unds.copy())
        K = tht(cam.K)

        optimizer = optim.Adam(param.values(), lr=3e-4)
        num_iterations = 14000
        for idx in range(num_iterations):
            lossd = {}
            optimizer.zero_grad()
            uv_unds_new = undistort(uv_ds, K, param["D"])
            retval = torch.sqrt(((uv_unds_new - uv_unds) ** 2).sum(-1) + eps).mean()
            # loss_l2 = ((uv_unds_new - uv_unds)**2).sum(-1).mean()

            loss_l1 = torch.abs(uv_unds_new - uv_unds).mean()
            loss_l2 = ((uv_unds_new - uv_unds) ** 2).mean()

            loss = loss_l2
            loss.backward()
            lossd["loss"] = loss
            if idx % 100 == 0:  # boxx.mg():
                # print({k: v.grad for k, v in param.items()})
                print(
                    f"Iteration {idx}: retval={strnum(retval.item())}, {', '.join([f'{k}={strnum(lossd[k])}' for k in lossd])}"
                )
            optimizer.step()
            # 1/0
            # [[-0.10747, 0.07222, 0.00051, 0.00011, -0.01407]]
            # [[-0.10748, 0.07221, 0.00051, 0.00011, -0.01406]]
    else:
        # [[-0.10747, 0.07222, 0.00051, 0.00011, -0.01407]]
        # [[-0.10773, 0.07195, 0.00077, -0.00015, -0.01432]]
        param = {
            "D": tht(cam.D * 0).requires_grad_(),
            "uv_unds": tht(uv_ds.copy()).requires_grad_(),
        }
        uv_ds = tht(uv_ds.copy())
        uv_unds_gt = tht(uv_unds.copy())
        uv_unds = param["uv_unds"]
        K = tht(cam.K)

        optimizer = optim.Adam(param.values(), lr=1e-2)
        # optimizer = optim.SGD(param.values(), lr=1e-6, momentum=0.95)
        num_iterations = 14000
        for idx in range(num_iterations):
            lossd = {}
            optimizer.zero_grad()
            uv_ds_new = distort(uv_unds, K, param["D"])
            # uv_unds_new = undistort(uv_ds, K, param["D"])
            retval = torch.sqrt(((uv_unds - uv_unds_gt) ** 2).sum(-1) + eps).mean()

            loss_l1 = (
                torch.abs(uv_ds_new - uv_ds).mean()
                + torch.abs(uv_unds - uv_unds_gt).mean()
            )
            loss_l2 = ((uv_ds_new - uv_ds) ** 2).mean() + (
                (uv_unds - uv_unds_gt) ** 2
            ).mean()

            loss = loss_l2
            loss.backward()
            lossd["loss"] = loss
            if idx % 100 == 0:  # boxx.mg():
                # print({k: v.grad for k, v in param.items()})
                print(
                    f"Iteration {idx}: retval={strnum(retval.item())}, {', '.join([f'{k}={strnum(lossd[k])}' for k in lossd])}"
                )
            # 1/0
            optimizer.step()
    print(cam.D.round(5).tolist(), f"\n{npa(param['D']).round(5).tolist()}")
