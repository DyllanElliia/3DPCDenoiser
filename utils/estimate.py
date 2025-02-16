'''
Author: DyllanElliia
Date: 2023-09-18 14:57:29
LastEditors: DyllanElliia
LastEditTime: 2023-09-18 20:08:43
Description: 
'''
import os
import torch
import pytorch3d
import numpy as np
import math

from sklearn.decomposition import PCA

from tqdm.auto import tqdm


# def pca_for_patchs(patchs):
#     N, k, d = patchs.shape
#     p_mean = torch.mean(patchs, axis=1)  # (N, 3)
#     p_mean = p_mean.unsqueeze(1)
#     print(patchs.shape, p_mean.shape)
#     patchs_m = patchs - p_mean  # (N, k ,3)
#     covariance_matrix = (
#         1 / k * torch.matmul(torch.transpose(patchs_m, 2, 1), patchs_m)
#     )  # (N, k, k)
#     eigenvectors, eigenvalues, _ = torch.svd(covariance_matrix)
#     print("eigen shape:", eigenvalues.shape, eigenvectors.shape)
#     # eigenvalues = torch.norm(eigenvalues, dim=1)
#     print(eigenvalues.shape)
#     idx = torch.argsort(eigenvalues, dim=1, descending=True).expand(N, 3, 3)
#     print("index", idx.shape)
#     # idx.unsqueeze(1)
#     # print("index", idx.shape)
#     print(idx[:5, :])
#     # eigenvectors = eigenvectors[idx]
#     eigenvectors = torch.gather(eigenvectors, dim=1, index=idx)
#     proj_mat = eigenvectors[:, :, :]
#     return p_mean, proj_mat


# def transform_patch(patchs, mean, proj):
#     patchs = patchs - mean
#     return patchs.matmul(proj)


def get_principle_dirs(pts):
    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs


def pca_alignment(pts, random_flag=False):
    pca_dirs = get_principle_dirs(pts)

    if random_flag:
        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))

    return pts, inv_rotation


def rotation_matrix(axis, theta):
    # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.matrix(
        np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
    )


def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if (
        np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0
        or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0
    ):
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [
        a[0] * a[0] * (1.0 - c) + c,
        a[0] * a[1] * (1.0 - c) - s * a[2],
        a[0] * a[2] * (1.0 - c) + s * a[1],
    ]

    R2 = [
        a[0] * a[1] * (1.0 - c) + s * a[2],
        a[1] * a[1] * (1.0 - c) + c,
        a[1] * a[2] * (1.0 - c) - s * a[0],
    ]

    R3 = [
        a[0] * a[2] * (1.0 - c) - s * a[1],
        a[1] * a[2] * (1.0 - c) + s * a[0],
        a[2] * a[2] * (1.0 - c) + c,
    ]

    R = np.matrix([R1, R2, R3])

    return R


def estimate_noise_sigma(pcl, k=1024, return_pca=False):
    N, d = pcl.shape
    pcl = pcl.unsqueeze(0)  # (1, N, 3)
    _, _, Patchs = pytorch3d.ops.knn_points(
        pcl, pcl, K=k, return_nn=True
    )  # (1, N, k, 3)
    Patchs = Patchs.squeeze(0)  # (N, k ,3)
    Patchs = Patchs.cpu().numpy()
    if return_pca:
        patch_pts = Patchs.copy()
        patch_pca = np.zeros_like(patch_pts)
    sigma = np.zeros(N, dtype=float)
    k2 = 1024
    for i in tqdm(range(N), desc='Estimate'):
        pts = Patchs[i, :, :]
        # print(pts.shape)
        pts, _ = pca_alignment(pts)
        if return_pca:
            patch_pts[i, :, :] = Patchs[i, :, :]
            patch_pca[i, :, :] = pts
        pts_z = pts[:k2, 2]
        # print(pts_z.shape)
        pts_dz = pts_z - np.mean(pts_z, axis=0)
        sigma_ = 1 / (k2 - 1) * np.sum(np.square(pts_dz), axis=0)
        sigma[i] = sigma_

        # print(sigma_)
    if return_pca:
        return sigma, patch_pts, patch_pca
    else:
        return np.mean(sigma) ** 0.5
