"""
Author: DyllanElliia
Date: 2022-12-14 11:21:58
LastEditors: DyllanElliia
LastEditTime: 2023-10-20 20:37:49
Description: 
"""
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList

from .schedule import VarianceSchedule

from .utils import farthest_point_sampling
import pytorch3d.ops

import numpy as np

import math

from einops import rearrange, repeat


def releaseGpuCache():
  torch.cuda.empty_cache()


# from .diffusion import VarianceSchedule
def split_tensor_to_segments(x, segsize):
  num_segs = math.ceil(x.size(0) / segsize)
  segs = []
  for i in range(num_segs):
    segs.append(x[i * segsize:(i + 1) * segsize])
  return segs


def grad_compute_u(grad_pred, x_t, Patchs_idx):
  """
    Args:
        grad_pred:  Input grad prediction, (B, N, K, 3)
        x_t: Point clouds, (B, N, 3)
        Patchs_idx: Point Patch indices, (B, N, K)
    """
  # return grad_pred.mean(2).squeeze(2)
  grad = grad_pred.reshape(x_t.size(0), -1, x_t.size(2))  # (B, N*K, 3)
  nn_idx = Patchs_idx.reshape(x_t.size(0), -1)  # (B, N*K,)
  nn_idx = nn_idx.unsqueeze(-1).expand_as(grad)  # (B, N*K, 1) -> (B, N*K, 3)

  # print(grad.shape, grad_pred.shape, nn_idx.shape)
  num_cnt = torch.ones_like(grad)
  num_grads = torch.zeros_like(x_t)
  num_grads.scatter_add_(dim=1, index=nn_idx, src=num_cnt)
  num_grads[num_grads < 1] = 1
  acc_grads = torch.zeros_like(x_t)
  acc_grads.scatter_add_(dim=1, index=nn_idx, src=grad)
  acc_grads = acc_grads / num_grads
  return acc_grads


def grad_compute(grad_pred, x_t, x_m, Patchs_idx, kappa):
  """
    Args:
        grad_pred:  Input grad prediction, (B, N, K, 3)
        x_t: Point clouds, (B, N, 3)
        Patchs_idx: Point Patch indices, (B, N, K)
    """
  # NOTE: single score

  # return torch.mean(grad_pred, dim=2)

  pi = np.pi
  # pdf_w = (kappa**2 + 1) / (2 * pi * (1 + np.exp(-kappa * pi)))
  u = grad_compute_u(grad_pred, x_t, Patchs_idx)

  if kappa == 0:
    return u

  hat_g = pytorch3d.ops.knn_gather(u, Patchs_idx)
  hat_g_norm = hat_g.norm(dim=-1, keepdim=True)
  gij = grad_pred
  gij_norm = grad_pred.norm(dim=-1, keepdim=True)

  dot_hg_gij = (gij * hat_g).sum(dim=-1).unsqueeze(dim=-1)

  cos_gu = torch.cos(dot_hg_gij / (gij_norm * hat_g_norm))
  pdf = torch.exp(-kappa / cos_gu)

  # angle = torch.acos(dot_hg_gij / (gij_norm * hat_g_norm)) + 1e-5
  # pdf = torch.exp(-kappa * angle)
  # pdf = torch.nan_to_num(pdf, nan=0.0)

  g_w = (grad_pred * pdf).reshape(x_t.size(0), -1, x_t.size(2))  # (B, N*K, 3)
  nn_idx = Patchs_idx.reshape(x_t.size(0), -1)  # (B, N*K,)
  nn_idx = nn_idx.unsqueeze(-1).expand_as(g_w)  # (B, N*K, 1) -> (B, N*K, 3)
  acc_g = torch.zeros_like(x_t)
  acc_g.scatter_add_(dim=1, index=nn_idx, src=g_w)

  pdf = pdf.expand_as(grad_pred).reshape(x_t.size(0), -1, x_t.size(2))
  acc_pdf = torch.zeros_like(x_t)
  acc_pdf.scatter_add_(dim=1, index=nn_idx, src=pdf)
  acc_pdf[acc_pdf < 1e-8] = 1
  acc_grad = acc_g / acc_pdf

  return acc_grad


def grad_compute_gauss(grad_pred, x_t, x_m, Patchs_idx, kappa, sigma):
  """
    Args:
        grad_pred:  Input grad prediction, (B, N, K, 3)
        x_t: Point clouds, (B, N, 3)
        Patchs_idx: Point Patch indices, (B, N, K)
    """
  # NOTE: single score

  # return torch.mean(grad_pred, dim=2)

  pi = np.pi
  # pdf_w = (kappa**2 + 1) / (2 * pi * (1 + np.exp(-kappa * pi)))
  u = grad_compute_u(grad_pred, x_t, Patchs_idx)

  if kappa == 0:
    return u

  hat_g = pytorch3d.ops.knn_gather(u, Patchs_idx)
  hat_g_norm = hat_g.norm(dim=-1, keepdim=True)
  gij = grad_pred
  gij_norm = grad_pred.norm(dim=-1, keepdim=True)

  dot_hg_gij = (gij * hat_g).sum(dim=-1).unsqueeze(dim=-1)

  cos_gu = torch.cos(dot_hg_gij / (gij_norm * hat_g_norm))
  pdf = torch.exp(-kappa / cos_gu)

  x_patch = pytorch3d.ops.knn_gather(x_t, Patchs_idx)  # (B, N, K, 3)
  dis_patch = x_patch - x_m.unsqueeze(2)  # (B, N, K, 3)
  dis2_patch = (dis_patch**2).sum(dim=-1, keepdim=True)  # (B, N, K, 1)
  max_dis, _ = torch.min(dis2_patch[:, :, -1, :], dim=1, keepdim=True)
  max_dis = torch.sqrt(max_dis).unsqueeze(dim=2)
  wi = (1 / (max_dis * np.sqrt(pi))) * torch.exp(-0.5 * dis2_patch /
                                                 (max_dis**2))
  # print(
  #     dis2_patch[0, 0, 0],
  #     (dis2_patch / sigma**2)[0, 0, 0],
  #     (torch.exp(-0.5 * dis2_patch / sigma**2))[0, 0, 0],
  #     ((1 / (sigma * np.sqrt(pi))) * torch.exp(-0.5 * dis2_patch / sigma**2))[
  #         0, 0, 0
  #     ],
  # )
  pdf = pdf * wi
  # angle = torch.acos(dot_hg_gij / (gij_norm * hat_g_norm)) + 1e-5
  # pdf = torch.exp(-kappa * angle)
  # pdf = torch.nan_to_num(pdf, nan=0.0)

  g_w = (grad_pred * pdf).reshape(x_t.size(0), -1, x_t.size(2))  # (B, N*K, 3)
  nn_idx = Patchs_idx.reshape(x_t.size(0), -1)  # (B, N*K,)
  nn_idx = nn_idx.unsqueeze(-1).expand_as(g_w)  # (B, N*K, 1) -> (B, N*K, 3)
  acc_g = torch.zeros_like(x_t)
  acc_g.scatter_add_(dim=1, index=nn_idx, src=g_w)

  pdf = pdf.expand_as(grad_pred).reshape(x_t.size(0), -1, x_t.size(2))
  acc_pdf = torch.zeros_like(x_t)
  acc_pdf.scatter_add_(dim=1, index=nn_idx, src=pdf)
  acc_pdf[acc_pdf < 1e-8] = 1
  acc_grad = acc_g / acc_pdf

  return acc_grad


def vs_sigma_n(b_n, num_steps, beta1, mode):
  s_n = []
  print(b_n.size)
  for i in range(b_n.size):
    vs = VarianceSchedule(
        num_steps=num_steps,
        beta_1=beta1,
        beta_T=b_n[i],
        mode=mode,
    )
    s_n.append(vs.vs_sigma())
  return s_n


def get_fio():
  Max_SIZE = 40
  fi = [0] * Max_SIZE
  fi[0] = 1
  fi[1] = 1
  for i in range(2, Max_SIZE):
    fi[i] = fi[i - 1] + fi[i - 2]
  return fi


def fio_find(nums, target):
  low, hight = 0, len(nums) - 1
  fi = get_fio()
  k = 0
  while hight > fi[k] - 1:
    k += 1
  temp = nums.copy()
  for i in range(hight + 1, fi[k]):
    temp.append(nums[hight])
  while low <= hight:
    mid = low + fi[k - 1] - 1
    # print(low, hight, mid)
    if temp[mid] < target:
      low = mid + 1
      k -= 2
    elif temp[mid] > target:
      hight = mid - 1
      k -= 1

    if mid <= hight:
      index = mid
    else:
      index = hight
  return index


class SBD(Module):

  def __init__(self, args, args_test=None):
    super().__init__()

    vs = VarianceSchedule(
        num_steps=args.num_steps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode=args.sched_mode,
    )
    if args.tag is not None and "2FF" in args.tag:
      # from .sbdiffusion_2FF0606 import DiffusionSB2FF
      print("use FF Diff v2 error!")
      # self.dsb = DiffusionSB2FF(args, var_sched=vs)
    elif args.tag is not None and "FFFF" in args.tag:
      from .sbdiffusion_2FF719 import DiffusionSB2FF
      print("use FF Diff with neural weight")
      self.dsb = DiffusionSB2FF(args, var_sched=vs)
    elif args.tag is not None and "FFF" in args.tag:
      from .sbdiffusion_2FF616 import DiffusionSB2FF
      print("use FF Diff with neural weight")
      self.dsb = DiffusionSB2FF(args, var_sched=vs)
    elif args.tag is not None and "FF" in args.tag:
      from .sbdiffusion_2FF528 import DiffusionSB2FF
      print("use FF Diff")
      self.dsb = DiffusionSB2FF(args, var_sched=vs)
    else:
      from .sbdiffusion import DiffusionSB
      self.dsb = DiffusionSB(args, var_sched=vs)

  def get_loss(self, x_0, t=None):
    return self.dsb.get_loss(x_0, t)

  def change_vs(self, args, noiseL=0.01, num_step=None):
    if num_step == None:
      num_step = args.num_steps
    print("do nothing")
    self.dsb.ctx_noiseL = noiseL
    self.dsb.ctx_num_steps = num_step
    return

    if args.beta_T > 0.0:
      vs = VarianceSchedule(
          num_steps=num_step,
          beta_1=args.beta_1,
          beta_T=args.beta_T,
          mode=args.sched_mode,
      )
    else:
      bT_l = args.beta_1
      while True:
        bT_r = bT_l * 1e4
        b_n = np.linspace(bT_l, bT_r, 10000)
        s_n = vs_sigma_n(b_n, num_step, args.beta_1, args.sched_mode)
        # i = fio_find(s_n, 1)
        i = fio_find(s_n, noiseL)
        if i >= 0:
          break
        else:
          print("Failed to find beta_T! max sigma is", s_n[-1])
          print("Change beta_1:", bT_l, "->", bT_l / 10)
          bT_l /= 10
      # print(b_n, s_n)
      vs = VarianceSchedule(
          num_steps=num_step,
          beta_1=bT_l,
          beta_T=b_n[i],
          mode=args.sched_mode,
      )
      print("beta_T:", b_n[i], i)
      print("sigma:", vs.vs_sigma())
      print("beta_1:", vs.betas[1])

    self.dsb.var_sched = vs

  def estimate_sigma(self, x_T, sample_knn=32, round=1, kappa=0):
    N = x_T.shape[0]
    if N > 100000:
      return self.estimate_sigma_big(x_T, sample_knn, 1, kappa)
    # round_s = 1 if N <= 100000 else int(np.ceil(N / 100000))
    # if round_s > 1:
    #     print("round:", round_s)
    x_T = x_T.unsqueeze(0)  # (1, N, 3)
    self.dsb.set_kappa(kappa)
    grad = self.dsb.get_grad_end2end(x_T, grad_compute, sample_knn, round)
    g_2 = torch.norm(grad.reshape(-1, 3), p=2, dim=1)
    n = g_2.shape[0]
    g_2[:n // 2] *= -1
    return torch.sqrt(torch.sum(g_2**2, dim=0) / (n - 1.0))

  def estimate_sigma_big(self, x_T, sample_knn=32, round=1, kappa=0):
    N = x_T.shape[0]

    x_T = x_T.unsqueeze(0)  # (1, N, 3)
    self.dsb.set_kappa(kappa)
    seed_pnts, _ = farthest_point_sampling(x_T, int(2 * N / 5000))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts,
                                             x_T,
                                             K=5000,
                                             return_nn=True)
    patches = patches[0]  # (N, K, 3)
    patches = split_tensor_to_segments(patches, 2)
    n = len(patches)
    print("so big... patch num:", n)
    patches_grad = []
    for i in range(n):
      patch_g = self.dsb.get_grad_end2end(patches[i], grad_compute, sample_knn,
                                          1)
      patches_grad.append(patch_g)
    grad = torch.cat(patches_grad, dim=0)
    # grad = self.dsb.get_grad_end2end(x_T, grad_compute, sample_knn, 1)
    g_2 = torch.norm(grad.reshape(-1, 3), p=2, dim=1)
    n = g_2.shape[0]
    g_2[:n // 2] *= -1
    return torch.sqrt(torch.sum(g_2**2, dim=0) / (n - 1.0))

  def estimate_sigma_score(self, x_T, sample_knn=32, round=10):
    x_T = x_T.unsqueeze(0)  # (1, N, 3)
    # self.dsb.set_kappa(kappa)
    score = self.dsb.get_score_end2end(x_T, grad_compute, sample_knn, round)
    g_2 = torch.norm(score.reshape(-1, 3), p=2, dim=1)
    n = g_2.shape[0]
    return torch.sqrt(torch.sum(g_2**2, dim=0) / (n - 1.0))

  def sample_end2end(self, x_noisy, gc, sample_knn, round=1):
    x_noisy = x_noisy.unsqueeze(0)  # (1, N, 3)
    self.dsb.set_kappa(0)
    score = self.dsb.get_grad_end2end(x_noisy, gc, sample_knn, round)
    return x_noisy + score

  def sample_end2end_niter(self,
                           x_noisy,
                           sample_knn,
                           T=2,
                           round=1,
                           kappa=0,
                           use_FF=True):
    x_noisy = x_noisy.unsqueeze(0)  # (1, N, 3)
    self.dsb.set_kappa(kappa)
    return self.dsb.sample_end2end(x_noisy,
                                   grad_compute,
                                   sample_knn,
                                   T,
                                   round,
                                   use_FF=use_FF)

  def sample_end2end_niter_patch(
      self,
      pcl_noisy,
      grad_compute_func,
      sample_knn,
      T=1,
      round=1,
      kappa=0,
      use_FF=True,
      seed_k=5,
      seed_k_alpha=2,
      patch_size=1000,
  ):
    # x_noisy = x_noisy.unsqueeze(0)  # (1, N, 3)
    # self.dsb.set_kappa(kappa)
    # return self.dsb.sample_end2end(
    #     x_noisy, grad_compute, sample_knn, T, round, use_FF=use_FF
    # )
    """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
    assert pcl_noisy.dim(
    ) == 2, "The shape of input point cloud must be (N, 3)."
    self.dsb.set_kappa(kappa)
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    num_patches = int(seed_k * N / patch_size)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(
        seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)

    # Patch stitching preliminaries
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
    patches = patches - seed_pnts_1
    patch_dists, point_idxs_in_main_pcd = patch_dists[
        0], point_idxs_in_main_pcd[0]
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(
        1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(
        point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd,
                                              patch_dists):
      all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)
    weights = torch.exp(-1 * all_dists)

    best_weights, best_weights_idx = torch.max(weights, dim=0)
    patches_denoised = []

    # Denoising
    i = 0
    patch_step = int(N / (seed_k_alpha * patch_size))
    assert (patch_step
            > 0), "Seed_k_alpha needs to be decreased to increase patch_step!"
    while i < num_patches:
      # print("Processed {:d}/{:d} patches.".format(i, num_patches))
      curr_patches = patches[i:i + patch_step]
      patches_denoised_temp = self.dsb.sample_end2end(
          curr_patches,
          grad_compute_func,
          sample_knn,
          T,
          round,
      )

      patches_denoised.append(patches_denoised_temp)
      i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_pnts_1

    # Patch stitching
    pcl_denoised = [
        patches_denoised[patch][point_idxs_in_main_pcd[patch] ==
                                pidx_in_main_pcd]
        for pidx_in_main_pcd, patch in enumerate(best_weights_idx)
    ]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)
    # print(pcl_denoised.shape)
    if pcl_denoised.shape[0] < N:
      print("miss point!:", pcl_denoised.shape)
      pcl_denoised = torch.cat(
          [
              pcl_denoised,
              torch.zeros(N - pcl_denoised.shape[0], d).to(pcl_denoised.device),
          ],
          dim=0,
      )

    return [pcl_denoised]

  def sample(
      self,
      x_T,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=32,
      var_sched=None,
      round=1,
      kappa=0,
      feature_fusion=True,
      is_sigma_eta_zero=True,
      eta=0,
      use_nearest_feat=False,
      break_at_step=1000,
      use_patch_base=False,
      alpha=0.6,
  ):
    N = x_T.shape[0]
    if N > 100000 or use_patch_base or (N == 100000 and sample_knn > 32):
      releaseGpuCache()
      releaseGpuCache()
      releaseGpuCache()
      seed_k = 8
      patch_size = 5000
      use_patch_scale = False
      if N >= 100000:
        print("big point cloud! v1")
        alpha = 3
      if N >= 150000:
        print("big point cloud! v1")
        alpha = 5
      if N > 300000:
        print("big point cloud! v2")
        alpha = 5
        seed_k = 5
        patch_size = 80000  # 50000 use in paper
        use_patch_scale = True
        # alpha = 25
        if sample_knn > 48:
          alpha = 15
        # alpha = 5
        # seed_k = 5
        # patch_size = 5000  # use in paper
        # use_patch_scale = True
        # # alpha = 25
        # if sample_knn > 48:
        #   alpha = 15

      if N < 9000:
        patch_size = 512

      return self.samplePatch_new(
          x_T,
          grad_compute,
          flexibility,
          ret_traj,
          sample_knn,
          var_sched,
          1,
          seed_k,
          alpha,
          patch_size,
          kappa,
          feature_fusion,
          is_sigma_eta_zero,
          eta,
          use_nearest_feat,
          use_patch_scale=use_patch_scale,
      )
    x_T = x_T.unsqueeze(0)  # (1, N, 3)
    self.dsb.set_kappa(kappa)
    return self.dsb.sample(
        x_T,
        grad_compute,
        flexibility,
        ret_traj,
        sample_knn,
        var_sched,
        round,
        feature_fusion,
        is_sigma_eta_zero,
        eta,
        use_nearest_feat,
        break_at_step,
    )

  def samplePatch(
      self,
      pcl_noisy,
      grad_compute_func=grad_compute,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=4,
      var_sched=None,
      round=1,
      seed_k=5,
      seed_k_alpha=2,
      patch_size=1000,
      kappa=0,
      feature_fusion=True,
      is_sigma_eta_zero=False,
      eta=0,
      use_nearest_feat=False,
  ):
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    _, N, d = pcl_noisy.size()

    seed_pnts, _ = farthest_point_sampling(x_T, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts,
                                             x_T,
                                             K=self.args.patch_size,
                                             return_nn=True)
    x_T = patches[0]  # (N, K, 3)
    res = self.dsb.sample(x_T, grad_compute, flexibility, ret_traj, sample_knn,
                          var_sched, round)
    if ret_traj:
      traj = res
      x0_denoised, fps_idx = farthest_point_sampling(traj[0].view(1, -1, d), N)
      x0_denoised = x0_denoised[0]
      fps_idx = fps_idx[0]

      for i in range(len(traj)):
        traj[i] = traj[i].view(-1, d)[fps_idx, :]
      return traj
    else:
      traj = res
      x0_denoised, fps_idx = farthest_point_sampling(traj.view(1, -1, d), N)
      x0_denoised = x0_denoised[0]
      return x0_denoised

  def samplePatch_new_fusion(
      self,
      pcl_noisy,
      grad_compute_func=grad_compute,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=4,
      var_sched=None,
      round=1,
      seed_k=5,
      seed_k_alpha=2,
      patch_size=1000,
      kappa=0,
      feature_fusion=True,
      is_sigma_eta_zero=False,
      eta=0,
      use_nearest_feat=False,
      use_patch_scale=False,
  ):
    """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
    assert pcl_noisy.dim(
    ) == 2, "The shape of input point cloud must be (N, 3)."
    print("------------------------------------------------------")
    print("patch args:")
    print("sample_knn:", sample_knn)
    print("seed_k:", seed_k)
    print("seed_k_alpha:", seed_k_alpha)
    print("patch_size:", patch_size)
    print("feature_fusion:", feature_fusion)
    print("is_sigma_eta_zero:", is_sigma_eta_zero)
    print("eta:", eta)
    print("------------------------------------------------------")
    self.dsb.set_kappa(kappa)
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    num_patches = int(seed_k * N / patch_size)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(
        seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)
    print("index", point_idxs_in_main_pcd.shape, patch_dists.shape)
    point_2_patch_index = point_idxs_in_main_pcd[0]
    # print("point_2_patch_index:", point_2_patch_index.shape)

    # Patch stitching preliminaries
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)

    def normalize_patch(patch, use_scale=False):
      # patch: (N, K, 3)
      patch = patch - seed_pnts_1
      if use_scale:
        p_scale = torch.norm(patch, p=2, dim=2,
                             keepdim=True).max(dim=1, keepdim=True)[0]
      else:
        p_scale = 1.0
      return patch / p_scale, p_scale

    def denormalize_patch(patch, p_scale):
      return patch * p_scale + seed_pnts_1

    # patches = patches - seed_pnts_1
    patches, p_scale = normalize_patch(patches, use_patch_scale)
    patch_dists, point_idxs_in_main_pcd = patch_dists[
        0], point_idxs_in_main_pcd[0]
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(
        1, patch_size)
    # patch_dists = patch_dists / torch.max(patch_dists.reshape(-1), dim=0)[0]
    print(patch_dists.shape)

    # all_dists = torch.ones(num_patches, N) / 0
    # all_dists = all_dists.cuda()
    # all_dists = list(all_dists)
    # patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(
    #     point_idxs_in_main_pcd)

    # for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd,
    #                                           patch_dists):
    #   all_dist[patch_id] = patch_dist

    # all_dists = torch.stack(all_dists, dim=0)
    # weights = torch.exp(-1 * all_dists)
    # weights = -1 * all_dists

    # best_weights, best_weights_idx = torch.max(weights, dim=0)
    patches_denoised = []

    # Denoising
    i = 0
    patch_step = int(N / (seed_k_alpha * patch_size))
    # if N <= 10000:
    #   patch_step = len(patches)
    # elif N <= 100000:
    #   patch_step = int(len(patches) / 2)
    # assert (patch_step
    #         > 0), "Seed_k_alpha needs to be decreased to increase patch_step!"
    if patch_step == 0:
      print("Seed_k_alpha needs to be decreased to increase patch_step!")
      patch_step = 1
    # print("patch num:", len(patches))
    print("patch num:", num_patches)
    print("patch step:", patch_step)
    while i < num_patches:
      print("Processed {:d}/{:d} patches.".format(i, num_patches))
      i_end = i + patch_step
      if i_end > num_patches:
        i_end = num_patches
      curr_patches = patches[i:i_end]
      patches_denoised_temp = self.dsb.sample(
          curr_patches,
          # grad_compute_func,
          flexibility=flexibility,
          ret_traj=ret_traj,
          sample_knn=sample_knn,
          var_sched=var_sched,
          round=round,
          feature_fusion=feature_fusion,
          is_sigma_eta_zero=is_sigma_eta_zero,
          eta=eta,
          use_nearest_feat=use_nearest_feat,
      )

      patches_denoised.append(patches_denoised_temp)
      i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)  # (P, K, 3)
    # patches_denoised = patches_denoised + seed_pnts_1
    patches_denoised = denormalize_patch(patches_denoised, p_scale)

    # Patch stitching
    # pcl_denoised = [
    #     patches_denoised[patch][point_idxs_in_main_pcd[patch] ==
    #                             pidx_in_main_pcd]
    #     for pidx_in_main_pcd, patch in enumerate(best_weights_idx)
    # ]
    # pcl_denoised = torch.cat(pcl_denoised, dim=0)
    def scatter_add(shape, index, src):
      res = torch.zeros(shape).to(src.device)
      for i in range(src.shape[0]):
        res[index[i]] += src[i]
        # print(res[index[i]].shape, src[i].shape)

      return res

    # pcl_denoised = torch.zeros(N, d).to(pcl_noisy.device)
    # patch_dists = torch.norm(patches_denoised - seed_pnts_1, p=2, dim=2)
    patch_dists = torch.norm(patches_denoised - patches_denoised[:, :1, :],
                             p=2,
                             dim=2)
    patch_dists /= patch_dists.max(dim=1, keepdim=True)[0]
    patch_dists = torch.exp(-9 * patch_dists)
    # patch_dists[patch_dists < 1e-8] = 0
    # gather the points in the same patch
    # print(patches_denoised.shape, point_2_patch_index.shape, patch_dists.shape)
    # point_idxs_in_main_pcd = torch.cat(point_idxs_in_main_pcd, dim=0)
    # print(point_idxs_in_main_pcd.shape)
    pcl_weight_ans = scatter_add((N), point_2_patch_index, patch_dists)
    # print(
    #     pcl_weight_ans.shape,
    #     pytorch3d.ops.knn_gather(repeat(pcl_weight_ans, "N -> 1 N 3"),
    #                              point_2_patch_index.unsqueeze(0)).shape)
    patch_dists = repeat(patch_dists, "N K -> N K 3")
    pcl_weight_ans = repeat(pcl_weight_ans, "N -> 1 N 3")
    pcl_weight_ans = pytorch3d.ops.knn_gather(
        pcl_weight_ans, point_2_patch_index.unsqueeze(0))[0]
    patch_dists = patch_dists / pcl_weight_ans
    pcl_denoised = scatter_add((N, d), point_2_patch_index,
                               patches_denoised * patch_dists)

    # return 1

    # print(pcl_denoised.shape)
    if pcl_denoised.shape[0] < N:
      print("miss point!:", pcl_denoised.shape)
      pcl_denoised = torch.cat(
          [
              pcl_denoised,
              pcl_denoised[0:N - pcl_denoised.shape[0]],
          ],
          dim=0,
      )
    return [pcl_denoised]

  def samplePatch_new_org(
      self,
      pcl_noisy,
      grad_compute_func=grad_compute,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=4,
      var_sched=None,
      round=1,
      seed_k=5,
      seed_k_alpha=2,
      patch_size=1000,
      kappa=0,
      feature_fusion=True,
      is_sigma_eta_zero=False,
      eta=0,
      use_nearest_feat=False,
      use_patch_scale=False,
  ):
    """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
    assert pcl_noisy.dim(
    ) == 2, "The shape of input point cloud must be (N, 3)."
    print("------------------------------------------------------")
    print("patch args:")
    print("sample_knn:", sample_knn)
    print("seed_k:", seed_k)
    print("seed_k_alpha:", seed_k_alpha)
    print("patch_size:", patch_size)
    print("feature_fusion:", feature_fusion)
    print("is_sigma_eta_zero:", is_sigma_eta_zero)
    print("eta:", eta)
    print("------------------------------------------------------")
    self.dsb.set_kappa(kappa)
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    num_patches = int(seed_k * N / patch_size)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(
        seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)

    # Patch stitching preliminaries
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)

    def normalize_patch(patch, use_scale=False):
      # patch: (N, K, 3)
      patch = patch - seed_pnts_1
      if use_scale:
        p_scale = torch.norm(patch, p=2, dim=2,
                             keepdim=True).max(dim=1, keepdim=True)[0]
      else:
        p_scale = 1.0
      return patch / p_scale, p_scale

    def denormalize_patch(patch, p_scale):
      return patch * p_scale + seed_pnts_1

    # patches = patches - seed_pnts_1
    patches, p_scale = normalize_patch(patches, use_patch_scale)
    patch_dists, point_idxs_in_main_pcd = patch_dists[
        0], point_idxs_in_main_pcd[0]
    patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(
        1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(
        point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd,
                                              patch_dists):
      all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)
    weights = torch.exp(-1 * all_dists)
    # weights = -1 * all_dists

    best_weights, best_weights_idx = torch.max(weights, dim=0)
    patches_denoised = []

    # Denoising
    i = 0
    patch_step = int(N / (seed_k_alpha * patch_size))
    # if N <= 10000:
    #   patch_step = len(patches)
    # elif N <= 100000:
    #   patch_step = int(len(patches) / 2)
    # assert (patch_step
    #         > 0), "Seed_k_alpha needs to be decreased to increase patch_step!"
    if patch_step == 0:
      print("Seed_k_alpha needs to be decreased to increase patch_step!")
      patch_step = 1
    # print("patch num:", len(patches))
    print("patch num:", num_patches)
    print("patch step:", patch_step)
    while i < num_patches:
      print("Processed {:d}/{:d} patches.".format(i, num_patches))
      i_end = i + patch_step
      if i_end > num_patches:
        i_end = num_patches
      curr_patches = patches[i:i_end]
      patches_denoised_temp = self.dsb.sample(
          curr_patches,
          # grad_compute_func,
          flexibility=flexibility,
          ret_traj=ret_traj,
          sample_knn=sample_knn,
          var_sched=var_sched,
          round=round,
          feature_fusion=feature_fusion,
          is_sigma_eta_zero=is_sigma_eta_zero,
          eta=eta,
          use_nearest_feat=use_nearest_feat,
      )

      patches_denoised.append(patches_denoised_temp)
      i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    # patches_denoised = patches_denoised + seed_pnts_1
    patches_denoised = denormalize_patch(patches_denoised, p_scale)

    # Patch stitching
    pcl_denoised = [
        patches_denoised[patch][point_idxs_in_main_pcd[patch] ==
                                pidx_in_main_pcd]
        for pidx_in_main_pcd, patch in enumerate(best_weights_idx)
    ]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)
    # print(pcl_denoised.shape)
    if pcl_denoised.shape[0] < N:
      print("miss point!:", pcl_denoised.shape)
      pcl_denoised = torch.cat(
          [
              pcl_denoised,
              pcl_denoised[0:N - pcl_denoised.shape[0]],
          ],
          dim=0,
      )
    return [pcl_denoised]

  def samplePatch_new(
      self,
      pcl_noisy,
      grad_compute_func=grad_compute,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=4,
      var_sched=None,
      round=1,
      seed_k=5,
      seed_k_alpha=2,
      patch_size=1000,
      kappa=0,
      feature_fusion=True,
      is_sigma_eta_zero=False,
      eta=0,
      use_nearest_feat=False,
      use_patch_scale=False,
  ):
    """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
    assert pcl_noisy.dim(
    ) == 2, "The shape of input point cloud must be (N, 3)."
    print("------------------------------------------------------")
    print("patch args:")
    print("sample_knn:", sample_knn)
    print("seed_k:", seed_k)
    print("seed_k_alpha:", seed_k_alpha)
    print("patch_size:", patch_size)
    print("feature_fusion:", feature_fusion)
    print("is_sigma_eta_zero:", is_sigma_eta_zero)
    print("eta:", eta)
    print("------------------------------------------------------")
    self.dsb.set_kappa(kappa)
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    num_patches = int(seed_k * N / patch_size)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(
        seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (N, K, 3)

    # Patch stitching preliminaries
    seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)

    def normalize_patch(patch, use_scale=False):
      # patch: (N, K, 3)
      patch = patch - seed_pnts_1
      if use_scale:
        p_scale = torch.norm(patch, p=2, dim=2,
                             keepdim=True).max(dim=1, keepdim=True)[0]
      else:
        p_scale = 1.0
      return patch / p_scale, p_scale

    def denormalize_patch(patch, p_scale):
      return patch * p_scale + seed_pnts_1

    # patches = patches - seed_pnts_1
    patches, p_scale = normalize_patch(patches, use_patch_scale)
    patch_dists, point_idxs_in_main_pcd = patch_dists[
        0], point_idxs_in_main_pcd[0]
    # patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(
    #     1, patch_size)

    all_dists = torch.ones(num_patches, N) / 0
    all_dists = all_dists.cuda()
    all_dists = list(all_dists)
    patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(
        point_idxs_in_main_pcd)

    for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd,
                                              patch_dists):
      all_dist[patch_id] = patch_dist

    all_dists = torch.stack(all_dists, dim=0)
    # weights = torch.exp(-1 * all_dists)
    weights = all_dists

    best_weights, best_weights_idx = torch.min(weights, dim=0)
    patches_denoised = []

    # Denoising
    i = 0
    patch_step = int(N / (seed_k_alpha * patch_size))
    # if N <= 10000:
    #   patch_step = len(patches)
    # elif N <= 100000:
    #   patch_step = int(len(patches) / 2)
    # assert (patch_step
    #         > 0), "Seed_k_alpha needs to be decreased to increase patch_step!"
    if patch_step == 0:
      print("Seed_k_alpha needs to be decreased to increase patch_step!")
      patch_step = 1
    # print("patch num:", len(patches))
    print("patch num:", num_patches)
    print("patch step:", patch_step)
    while i < num_patches:
      print("Processed {:d}/{:d} patches.".format(i, num_patches))
      i_end = i + patch_step
      if i_end > num_patches:
        i_end = num_patches
      curr_patches = patches[i:i_end]
      patches_denoised_temp = self.dsb.sample(
          curr_patches,
          # grad_compute_func,
          flexibility=flexibility,
          ret_traj=ret_traj,
          sample_knn=sample_knn,
          var_sched=var_sched,
          round=round,
          feature_fusion=feature_fusion,
          is_sigma_eta_zero=is_sigma_eta_zero,
          eta=eta,
          use_nearest_feat=use_nearest_feat,
      )

      patches_denoised.append(patches_denoised_temp)
      i += patch_step

    patches_denoised = torch.cat(patches_denoised, dim=0)
    # patches_denoised = patches_denoised + seed_pnts_1
    patches_denoised = denormalize_patch(patches_denoised, p_scale)

    # Patch stitching
    pcl_denoised = [
        patches_denoised[patch][point_idxs_in_main_pcd[patch] ==
                                pidx_in_main_pcd]
        for pidx_in_main_pcd, patch in enumerate(best_weights_idx)
    ]

    pcl_denoised = torch.cat(pcl_denoised, dim=0)
    # print(pcl_denoised.shape)
    if pcl_denoised.shape[0] < N:
      print("miss point!:", pcl_denoised.shape)
      pcl_denoised = torch.cat(
          [
              pcl_denoised,
              pcl_denoised[0:N - pcl_denoised.shape[0]],
          ],
          dim=0,
      )
    return [pcl_denoised]
