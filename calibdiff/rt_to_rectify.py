#!/usr/bin/env python3
import boxx, torch
from boxx import np

with boxx.inpkg():
    from . import calibdiff_utils

eps = 1e-8


def project_vec_on_plane(v, plane_v):
    return v - torch.dot(v, plane_v) / (torch.norm(plane_v) ** 2 + eps) * plane_v


def rotate_shortest_of_two_vecs(v1, v2):
    cross = torch.cross(v1, v2)
    rad = torch.acos(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + eps))
    r = rad * cross / (torch.norm(cross) + eps)

    return calibdiff_utils.rodrigues_pytorch(r)


def stereo_recitfy(R, t):
    axes_z = torch.tensor([0, 0, 1.0]).type_as(R)
    axes_nx = torch.tensor([-1.0, 0, 0]).type_as(R)
    t = t.squeeze()
    plane_v = t
    z_on_plane2 = project_vec_on_plane(axes_z, plane_v)
    z_on_plane1 = project_vec_on_plane(R @ axes_z, plane_v)
    z_on_plane = z_on_plane2 / torch.norm(z_on_plane2) + z_on_plane1 / torch.norm(
        z_on_plane1
    )

    R_align_x = rotate_shortest_of_two_vecs(axes_nx, t)
    R_align_z = rotate_shortest_of_two_vecs(R_align_x @ axes_z, z_on_plane)

    R2 = (R_align_z @ R_align_x).T
    R1 = R2 @ R[:3, :3]
    return dict(R1=R1, R2=R2)


if __name__ == "__main__":
    from boxx.ylth import *
    import calibrating

    d = calibdiff_utils.get_test_data()

    R, t = d["R"], d["t"]
    re = stereo_recitfy(R, t)
    gt = dicto(R=npa(R), t=npa(t))
    calibrating.Stereo.stereo_recitfy(gt)
    print(re["R1"])
    print(gt["R1"])
    print(re["R2"])
    print(gt["R2"])

    loss = (re["R1"] + re["R2"]).sum()
    loss.backward()
    print(d["t"].grad, d["r"].grad)


"""
# Convert to PyTorch code, note that there is a funcation name rodrigues_pytorch are equl to cv2.Rodrigues(r)[0]

def project_vec_on_plane(v, plane_v):
    return v - np.dot(v, plane_v) / (np.linalg.norm(plane_v) ** 2) * plane_v


def rotate_shortest_of_two_vecs(v1, v2, return_rodrigues=False):
    cross = np.cross(v1, v2)
    rad = np.arccos((v1 * v2).sum() / np.linalg.norm(v1) / np.linalg.norm(v2))
    r = rad * cross / (np.linalg.norm(cross) + eps)

    return cv2.Rodrigues(r)[0]

def stereo_recitfy(R, t):
    axes_z = np.array([0, 0, 1.0])
    axes_nx = np.array([-1.0, 0, 0])
    t = t.squeeze()
    plane_v = t
    z_on_plane2 = project_vec_on_plane(axes_z, plane_v)
    z_on_plane1 = project_vec_on_plane(R @ axes_z, plane_v)
    z_on_plane = z_on_plane2 / np.linalg.norm(
        z_on_plane2
    ) + z_on_plane1 / np.linalg.norm(z_on_plane1)

    R_align_x = rotate_shortest_of_two_vecs(axes_nx, t)
    R_align_z = rotate_shortest_of_two_vecs(R_align_x @ axes_z, z_on_plane)
    
    R2 = (R_align_z @ R_align_x).T
    R1 = R2 @ R[:3, :3])
    return R2, R1
    """
