"""
Author: DyllanElliia
Date: 2023-06-08 16:03:54
LastEditors: DyllanElliia
LastEditTime: 2023-10-22 18:30:10
Description: 
"""
"""
Author: DyllanElliia
Date: 2022-12-12 14:47:50
LastEditors: DyllanElliia
LastEditTime: 2023-06-08 16:03:54
Description: 
"""
import torch
import torch.nn.functional as F
from typing import Union
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
import time
from einops import rearrange, repeat
from .schedule import VarianceSchedule

# from .common import *

# from .diffusion import VarianceSchedule

# from .score import ScoreNet

# # from .feature import FeatureExtraction
# from .encoders.denseDgcnn import FeatureExtraction

from .utils import farthest_point_sampling


def get_random_indices(n, m):
  # print(m, n)
  # assert m <= n
  if m >= n:
    return np.arange(n)
  return np.random.permutation(n)[:m]


def infnanCheck(name, value):
  print(name + "inf", torch.isinf(value).any())
  print(name + "nan", torch.isnan(value).any())


def releaseGpuCache():
  torch.cuda.empty_cache()


#   time.sleep(1)


def rf_Tt_sb(T, v):
  T = np.array(T).astype(float)
  T = (0.2 * v.num_steps + 2.8 * T) / 3.0
  T = T.astype(int).tolist()
  return T


def rf_Tt_nt(T, v):
  # T = np.array(T).astype(float)
  # T = (0.2 * v.num_steps + 2.8 * T) / 3.0
  # T = T.astype(int).tolist()
  return T


# def knn(ref, query, k):
#     ref_c =torch.stack([ref] * query.shape[-1], dim=0).permute(0, 2, 1).reshape(-1, 2).transpose(0, 1)
#     query_c = torch.repeat_interleave(query, repeats=ref.shape[-1], dim=1)
#     delta = query_c - ref_c
#     distances = torch.sqrt(torch.pow(delta, 2).sum(dim=0))
#     distances = distances.view(query.shape[-1], ref.shape[-1])
#     sorted_dist, indices = torch.sort(distances, dim=-1)
#     return sorted_dist[:, :k], indices[:, :k]


class DiffusionSB2FF(Module):

  def __init__(self, args, var_sched: VarianceSchedule):
    super().__init__()
    self.args = args
    # geometry
    self.frame_knn = args.frame_knn
    # self.sample_knn = args.sample_knn
    self.num_train_points = args.num_train_points
    self.num_clean_nbs = args.num_clean_nbs
    if hasattr(args, "patch_ratio"):
      self.patch_ratio = args.patch_ratio
    else:
      self.patch_ratio = 1.2
    if hasattr(args, "num_selfsup_nbs"):
      self.num_selfsup_nbs = args.num_selfsup_nbs
    if hasattr(args, "use_sigmaLoss"):
      self.use_sigmaLoss = args.use_sigmaLoss
    else:
      self.use_sigmaLoss = False

    if self.use_sigmaLoss:
      print("use new loss")
    else:
      print("use old loss")
    # score-matching
    self.dsm_sigma = args.dsm_sigma
    # networks
    if hasattr(args, "edge_conv_knn"):
      self.edge_conv_knn = args.edge_conv_knn
    else:
      self.edge_conv_knn = 16
    if hasattr(args, "conv_num_fc_layers"):
      self.conv_num_fc_layers = args.conv_num_fc_layers
    else:
      self.conv_num_fc_layers = 3
    if hasattr(args, "conv_growth_rate"):
      self.conv_growth_rate = args.conv_growth_rate
    else:
      self.conv_growth_rate = 12
    if hasattr(args, "num_convs"):
      self.num_convs = args.num_convs
    else:
      self.num_convs = 4
    if hasattr(args, "conv_channels"):
      self.conv_channels = args.conv_channels
    else:
      self.conv_channels = 24
    if hasattr(args, "mlp_feat"):
      if args.mlp_feat:
        print("use mlp")
      else:
        print("not use mlp")
      self.mlp_feat = args.mlp_feat
    else:
      print("not use mlp")
      self.mlp_feat = False
    if hasattr(args, "cat_feat"):
      if args.cat_feat:
        print("use cat")
      else:
        print("not use cat")
      self.cat_feat = args.cat_feat
    else:
      print("not use cat")
      self.cat_feat = False

    # from .feature import FeatureExtraction
    if args.tag == "FF0319":
      from .encoders.denseDgcnn import FeatureExtraction

      self.featureNet = FeatureExtraction(
          conv_knn=self.edge_conv_knn,
          conv_num_fc_layers=self.conv_num_fc_layers,
          conv_growth_rate=self.conv_growth_rate,
          num_convs=self.num_convs,
          conv_channels=self.conv_channels,
          mlp_feat=self.mlp_feat,
          cat_feat=self.cat_feat,
      )
    elif args.tag == "FF0323":
      from .encoders.Dgcnn_tg import FeatureExtraction

      self.featureNet = FeatureExtraction(
          conv_knn=self.edge_conv_knn,
          conv_num_fc_layers=self.conv_num_fc_layers,
          num_convs=self.num_convs,
      )
    else:
      from .encoders.denseDgcnn import FeatureExtraction

      self.featureNet = FeatureExtraction(
          conv_knn=self.edge_conv_knn,
          conv_num_fc_layers=self.conv_num_fc_layers,
          conv_growth_rate=self.conv_growth_rate,
          num_convs=self.num_convs,
          conv_channels=self.conv_channels,
          mlp_feat=self.mlp_feat,
          cat_feat=self.cat_feat,
      )
    self.featureOut_channels = self.featureNet.out_channels
    # if hasattr(args, "use_1dGausssE"):
    #     self.use_1dGausssE = args.use_1dGausssE
    # else:
    #     self.use_1dGausssE = True
    if hasattr(args, "score_net_hidden_h"):
      print("use hidden h")
      self.score_net_hidden_h = args.score_net_hidden_h
    else:
      print("not use hidden h")
      self.score_net_hidden_h = args.score_net_hidden_dim
    if hasattr(args, "score_net_decoder_h"):
      print("use decoder h")
      self.score_net_decoder_h = args.score_net_decoder_h
      if self.score_net_decoder_h == 0:
        self.score_net_decoder_h = None
    else:
      print("not use decoder h")
      self.score_net_decoder_h = None
    print("Feature_channels:", self.featureOut_channels)

    # from .score import ScoreNet
    if args.tag == "0319":
      from .decoders.score_0319 import ScoreNet

      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
    elif args.tag == "0527":
      from .decoders.score_0527 import ScoreNet

      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
    else:
      from .decoders.score import ScoreNet

      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          hidden_h=self.score_net_hidden_h,
          num_blocks=args.score_net_num_blocks,
          decoder_h=self.score_net_decoder_h,
      )
    # self.loss_use_displacement = args.loss_use_displacement
    # self.displacement_alpha = args.displacement_alpha
    if args.Tgen_mode == "SB":
      print("Other:", args.Tgen_mode)
      self.randomFunc_Tt = rf_Tt_sb
    else:
      print("do no thing.", args.Tgen_mode)
      self.randomFunc_Tt = rf_Tt_nt

    if hasattr(args, "use_patch_ratio"):
      self.use_patch_ratio = args.use_patch_ratio
    else:
      self.use_patch_ratio = False
    if hasattr(args, "use_eqdis"):
      self.use_eqdis = args.use_eqdis
    else:
      self.use_eqdis = False
    if hasattr(args, "alpha"):
      self.alpha_mix = args.alpha
    else:
      self.use_eqdis = False
    if hasattr(args, "loss_gauss_weight"):
      self.loss_gauss_weight = args.loss_gauss_weight
    else:
      self.loss_gauss_weight = False
    # variables (for training)
    self.var_sched = var_sched
    self.train_sigma=self.var_sched.

  def get_loss(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    batch_size, point_num, point_dim = x_0.size()
    if self.use_eqdis:
      t = self.var_sched.eqdis_uniform_sample_t(batch_size)
      # T = self.var_sched.eqdis_uniform_sample_t(batch_size)
    else:
      t = self.var_sched.uniform_sample_t(batch_size)
      # T = self.var_sched.uniform_sample_t(batch_size)
    # DEBUG: try to remove it.
    t = self.randomFunc_Tt(t, self.var_sched)
    # alpha_bar_T = self.var_sched.alpha_bars[T]
    # beta_T = self.var_sched.betas[T]
    alpha_bar_t = self.var_sched.alpha_bars[t]
    # t_1 = [t[i] - 1 for i in range(batch_size)]
    # alpha_bar_tl = self.var_sched.alpha_bars[t_1]
    # beta_t = self.var_sched.betas[t]
    # alpha_t = self.var_sched.alphas[t]

    # sa_b_T = torch.sqrt(alpha_bar_T).view(-1, 1, 1)  # (B, 1, 1)
    # sa_b_i_T = torch.sqrt(1 - alpha_bar_T).view(-1, 1, 1)  # (B, 1, 1)
    sa_b_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)
    sa_b_i_t = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)
    # sa_t = torch.sqrt(alpha_t).view(-1, 1, 1)
    # sa_i_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1)
    # b_t = beta_t.clone().detach().view(-1, 1, 1)
    # print(b_t.shape, len(beta_t), len(alpha_t))
    t=torch.tensor(t).to(x_0.device)
    tT=rearrange(t,"B -> B () ()")/self.var_sched.num_steps
    # print(t.shape)
    
    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    e_rand = 0.03*torch.randn_like(x_0)  # (B, N, d)
    # NOTE: evaluate GuassianNoise
    x_t_a = x_0 + tT * e_rand

    # if self.use_1dGausssE:
    #     _, _, x0_nearest = pytorch3d.ops.knn_points(x_t_a, x_0, K=1, return_nn=True)
    #     x0_nearest = x0_nearest.view(batch_size, point_num, point_dim)
    #     e_rand = (x_t_a - x0_nearest) * sa_b_t / sa_b_i_t
    #     x_t_a = (sa_b_t * x0_nearest + sa_b_i_t * e_rand) / sa_b_t
    # else:
    #     x_t_a = (sa_b_t * x_0 + sa_b_i_t * e_rand) / sa_b_t

    feat = self.featureNet(x_t_a)  # (B, N, F)
    # make point cloud smaller
    # xt_p_idx = get_random_indices(
    #     int(point_num / self.patch_ratio), self.num_train_points
    # )
    if self.use_patch_ratio:
      xt_p_idx = get_random_indices(int(point_num / self.patch_ratio),
                                    self.num_train_points)
    else:
      xt_p_idx = get_random_indices(point_num, self.num_train_points)
    feat = feat[:, xt_p_idx, :]  # (B, n, F)
    F_n = feat.size(-1)
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]

    # NOTE: Local Patch construction
    _, _, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)

    # NOTE: use Score-Based compute grad
    grad_pred = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=feat.view(-1, F_n),
    ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
              point_dim)  # (B, n, K, 3)

    # NOTE: Local Patch construction for Clear Point
    _, _, clean_Patchs = pytorch3d.ops.knn_points(
        Patchs.view(-1, self.frame_knn, point_dim),  # (B*n, K, 3)
        x_0.unsqueeze(1).repeat(1, len(xt_p_idx), 1,
                                1).view(-1, point_num,
                                        point_dim),  # (B*n, M, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B*n, K, C, 3)
    clean_Patchs = clean_Patchs.view(batch_size, len(xt_p_idx), self.frame_knn,
                                     self.num_clean_nbs,
                                     point_dim)  # (B, n, K, C, 3)
    # NOTE: compute target vectors
    gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    gt_target = -1 * gt_offset

    if self.use_sigmaLoss:
      sig2 = ((self.alpha_mix +
               (1 - self.alpha_mix) * sa_b_t / sa_b_i_t)**2).view(-1, 1, 1, 1)
    else:
      sig2 = 1.0

    loss = (sig2 * (grad_pred - gt_target)**2.0).sum(
        dim=-1).mean() * (1.0 / self.dsm_sigma)

    # if self.loss_use_displacement:
    #     x0_use_xt_patch_index = pytorch3d.ops.knn_gather(x_0, Patchs_idx)
    #     # alpha=self.displacement_alpha*
    #     loss += self.displacement_alpha * (
    #         ((Patchs - grad_pred - x0_use_xt_patch_index) ** 2.0).sum(dim=-1).mean()
    #     )
    return loss

  def SB_GridEstimate(self,
                      x_T_a,
                      x_m_a,
                      x_t_a,
                      feat,
                      B,
                      N,
                      K,
                      d,
                      F,
                      round=10,
                      use_nearest_feat=False):
    index = [list(range(0 + i, N, round)) for i in range(round)]
    grad_pred = torch.zeros((B, N, K, d), device=x_T_a.device)
    Patchs_idx = torch.zeros((B, N, K), dtype=torch.int64, device=x_T_a.device)
    if use_nearest_feat:
      print("use nearest feat 1")
      for x_r_i in index:
        # x_T_a_r_i = x_T_a[:, x_r_i, :]
        x_t_a_r_i = x_t_a[:, x_r_i, :]
        # x_m_a_r_i = x_m_a[:, x_r_i, :]
        _, mi, _ = pytorch3d.ops.knn_points(x_t_a_r_i,
                                            x_t_a,
                                            K=1,
                                            return_nn=True)  # (B, N, k, 3)
        # print(x_m_a_r_i.shape)
        x_m_a_r_i = pytorch3d.ops.knn_gather(x_m_a, mi)
        # x_m_a_r_i = x_m_a_r_i.squeeze(2)
        x_m_a_r_i = torch.mean(x_m_a_r_i, dim=2)
        # x_m_a_r_i = x_m_a[:, x_r_i, :]
        Ni = len(x_r_i)
        _, pi, Patchs = pytorch3d.ops.knn_points(x_m_a_r_i,
                                                 x_t_a,
                                                 K=K,
                                                 return_nn=True)  # (B, N, K, 3)
        # print(type(pi), pi[0, 0, 0],type(pi[0, 0, 0]))
        Patchs_idx[:, x_r_i, :] = pi

        x_T_a_r_i = pytorch3d.ops.knn_gather(x_T_a, mi)
        # x_T_a_r_i = torch.mean(pytorch3d.ops.knn_gather(x_T_a, mi), dim=2).unsqueeze(2)
        # x_T_a_r_i = x_T_a[:, x_r_i, :].unsqueeze(2)
        # Patchs_center = Patchs - x_T_a_r_i  # (B, Ni, K, 3)
        # print(Patchs.shape, x_T_a_r_i.shape, Patchs_center.shape)
        Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)

        # feat_m = pytorch3d.ops.knn_gather(feat, mi)
        feat_m = torch.mean(pytorch3d.ops.knn_gather(feat, mi), dim=2)
        grad_pred[:, x_r_i, :, :] = self.scoreNet(
            x=Patchs_center.view(-1, K, d),  # (B*Ni, K, 3)
            c=feat_m.view(-1, F),  # (B*Ni, F)
        ).reshape(B, Ni, K, d)  # (B, Ni, K, 3)
      # for x_r_i in index:
      #     x_T_a_r_i = x_T_a[:, x_r_i, :]
      #     # x_t_a_r_i = x_t_a[:, x_r_i, :]
      #     _, mi, x_m_a_r_i = pytorch3d.ops.knn_points(
      #         x_T_a_r_i, x_m_a, K=4, return_nn=True
      #     )  # (B, N, k, 3)
      #     # print(x_m_a_r_i.shape)
      #     # x_m_a_r_i = x_m_a_r_i.squeeze(2)
      #     x_m_a_r_i = torch.mean(x_m_a_r_i, dim=2)
      #     # x_m_a_r_i = x_m_a[:, x_r_i, :]
      #     Ni = len(x_r_i)
      #     _, pi, Patchs = pytorch3d.ops.knn_points(
      #         x_m_a_r_i, x_t_a, K=K, return_nn=True
      #     )  # (B, N, K, 3)
      #     # print(type(pi), pi[0, 0, 0],type(pi[0, 0, 0]))
      #     Patchs_idx[:, x_r_i, :] = pi

      #     x_T_a_r_i = pytorch3d.ops.knn_gather(x_T_a, mi)
      #     # x_T_a_r_i = torch.mean(pytorch3d.ops.knn_gather(x_T_a, mi), dim=2).unsqueeze(2)
      #     # x_T_a_r_i = x_T_a[:, x_r_i, :].unsqueeze(2)
      #     # Patchs_center = Patchs - x_T_a_r_i  # (B, Ni, K, 3)
      #     # print(Patchs.shape, x_T_a_r_i.shape, Patchs_center.shape)
      #     Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)

      #     # feat_m = pytorch3d.ops.knn_gather(feat, mi)
      #     feat_m = torch.mean(pytorch3d.ops.knn_gather(feat, mi), dim=2)
      #     grad_pred[:, x_r_i, :, :] = self.scoreNet(
      #         x=Patchs_center.view(-1, K, d),  # (B*Ni, K, 3)
      #         c=feat_m.view(-1, F),  # (B*Ni, F)
      #     ).reshape(
      #         B, Ni, K, d
      #     )  # (B, Ni, K, 3)
      return grad_pred, Patchs_idx
    else:
      for x_r_i in index:
        x_T_a_r_i = x_T_a[:, x_r_i, :]
        x_m_a_r_i = x_m_a[:, x_r_i, :]
        Ni = len(x_r_i)
        _, pi, Patchs = pytorch3d.ops.knn_points(x_m_a_r_i,
                                                 x_t_a,
                                                 K=K,
                                                 return_nn=True)  # (B, N, K, 3)
        # print(type(pi), pi[0, 0, 0],type(pi[0, 0, 0]))
        Patchs_idx[:, x_r_i, :] = pi
        # Patchs_center = Patchs - x_T_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)
        Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)

        grad_pred[:, x_r_i, :, :] = self.scoreNet(
            x=Patchs_center.view(-1, K, d),  # (B*Ni, K, 3)
            c=feat[:, x_r_i, :].view(-1, F),  # (B*Ni, F)
        ).reshape(B, Ni, K, d)  # (B, Ni, K, 3)
      return grad_pred, Patchs_idx

  def SB_single_GridEstimate(self,
                             x_T_a,
                             x_m_a,
                             x_t_a,
                             feat,
                             B,
                             N,
                             d,
                             F,
                             round=10):
    x_c = x_t_a - x_T_a  # (B, N, 3)
    return self.scoreNet(
        x=x_c.view(-1, 1, d),
        c=feat.view(-1, F)  # (B*N, 1, 3)  # (B*N, F)
    ).reshape(B, N, d)

  def set_kappa(self, kappa: float):
    self.kappa = kappa

  def get_score_end2end(
      self,
      x_T,
      grad_compute_func,
      sample_knn=64,
      round=10,
  ):
    batch_size, num_points, point_dim = x_T.size()
    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      F_n = feat_T.size(-1)
      score, Patchs_idx = self.SB_GridEstimate(
          x_T,
          x_T,
          x_T,
          feat_T,
          batch_size,
          num_points,
          sample_knn,
          point_dim,
          F_n,
          round,
      )

    return score

  def get_grad_end2end(
      self,
      x_T,
      grad_compute_func,
      sample_knn=64,
      round=1,
  ):
    batch_size, num_points, point_dim = x_T.size()
    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      F_n = feat_T.size(-1)
      grad_pred, Patchs_idx = self.SB_GridEstimate(
          x_T,
          x_T,
          x_T,
          feat_T,
          batch_size,
          num_points,
          sample_knn,
          point_dim,
          F_n,
          round,
      )

      grad = grad_compute_func(grad_pred, x_T, x_T, Patchs_idx, self.kappa)
    return grad

  # org
  def sample(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
      feature_fusion=True,
      is_sigma_eta_zero=True,
      use_nearest_feat=False,
      break_at_step=1000,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}
    mu = {}

    mt = int(var_sched.num_steps / 2)

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        # TODO: rand it with the
        # z = (
        #     torch.randn(batch_size, num_points, 1).to(device)
        #     if t > 1
        #     else torch.zeros(batch_size, num_points, 1).to(device)
        # )

        # alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        alpha = var_sched.alphas[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        # sa = torch.sqrt(alpha)
        # b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        # NOTE:　w-sum
        # w = sa_b_i / sa_b / noiseL
        # feat_acc += feat_t_ * w
        # w_cnt += w
        # feat_t = (1 - w_cnt) * feat_T + feat_acc
        # x_m = (1 - w_cnt) * x_T + w_cnt * x_t

        # NOTE:　w-mean
        # w = sa_b_i / sa_b / noiseL
        # feat_acc += feat_t_ * w
        # w_cnt += w
        # x_acc_a += x_t * w
        # feat_t = feat_acc / w_cnt
        # x_m = x_acc_a / w_cnt
        # print("mean")
        # if t == break_at_step:
        #     print("do break")
        #     # feat_acc = torch.zeros_like(feat_acc)
        #     # x_acc_a = torch.zeros_like(x_acc_a)
        #     # w_cnt = 0
        #     traj[0] = x_t.clone()
        #     break
        # NOTE:　mean
        if feature_fusion:
          w = 1
          feat_acc += feat_t_ * w
          w_cnt += w
          x_acc_a += x_t * w
          feat_t = feat_acc / w_cnt
          x_m = x_acc_a / w_cnt  # (B, N, 3)
        else:
          feat_t = feat_T
          x_m = x_T
        # print("mean")

        # NOTE:　max-pool
        # feat_t = torch.max(feat_t, feat_t_)
        # # w = sa_b_i / sa_b / noiseL
        # # w_cnt += w

        # # x_m = (1 - w_cnt) * x_T + w_cnt * x_t
        # x_m = torch.max(x_t, x_acc_a)
        # x_acc_a = x_m.clone()

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
            use_nearest_feat,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa)
        # e_theta = -1 * grad * sa_b / sa_b_i

        if is_sigma_eta_zero:
          mu_w = 1 - torch.nan_to_num(
              torch.sqrt((alpha - alpha_bar) / (1 - alpha_bar)),
              nan=0,
              posinf=0,
          )
          print("eta=0: ", mu_w)
          x_next = x_t + mu_w * grad

        else:
          mu_w = torch.nan_to_num(beta / (1 - alpha_bar), nan=0, posinf=0)
          mu_t = x_t + mu_w * grad
          # x_next = mu_t
          z = (torch.randn(batch_size, num_points) if t > 1 else torch.zeros(
              batch_size, num_points)).to(device)
          e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
          e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                               e_norm)
          x_next = mu_t + sigma * z.unsqueeze(dim=-1) * e_norm
          # z = (
          #     torch.randn(batch_size, num_points, 3)
          #     if t > 1
          #     else torch.zeros(batch_size, num_points, 3)
          # ).to(device)

          # x_next = mu_t + sigma * z
        # print(t, "-> mu:", mu_w, "; sigma:", sigma, "; xt[1]:", x_next[0, 1])

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        else:
          mu[t] = mu_t.clone().cpu()
        # traj[0] = x_T_a + grad
        # break
        # if t == break_at_step:
        #     print("do break")
        #     # feat_acc = torch.zeros_like(feat_acc)
        #     # x_acc_a = torch.zeros_like(x_acc_a)
        #     # w_cnt = 0
        #     traj[0] = x_next.detach()
        #     break
    if ret_traj:
      return traj, mu
    else:
      return traj[0]

  # org
  def sample_eta(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
      feature_fusion=True,
      eta=0,
      use_nearest_feat=False,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}
    mu = {}

    mt = int(var_sched.num_steps / 2)

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        alpha_bar = var_sched.alpha_bars[t]
        alpha = var_sched.alphas[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        # sa = torch.sqrt(alpha)
        # b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        if feature_fusion:
          w = 1
          feat_acc += feat_t_ * w
          w_cnt += w
          x_acc_a += x_t * w
          feat_t = feat_acc / w_cnt
          x_m = x_acc_a / w_cnt  # (B, N, 3)
        else:
          feat_t = feat_T
          x_m = x_T

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
            use_nearest_feat,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa)
        # e_theta = -1 * grad * sa_b / sa_b_i

        mu_w = torch.nan_to_num(
            1 - torch.sqrt(
                (alpha - alpha_bar * (1 + eta * sigma**2)) / (1 - alpha_bar)),
            nan=0,
            posinf=0,
        )
        mu_t = x_t + mu_w * grad
        # x_next = mu_t
        # z = (
        #     torch.randn(batch_size, num_points)
        #     if t > 1
        #     else torch.zeros(batch_size, num_points)
        # ).to(device)
        # e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        # e_norm = torch.where(
        #     torch.isnan(e_norm), torch.full_like(e_norm, 0), e_norm
        # )
        # x_next = mu_t + np.sqrt(eta) * sigma * z.unsqueeze(dim=-1) * e_norm
        z = (torch.randn(batch_size, num_points, 3) if t > 1 else torch.zeros(
            batch_size, num_points, 3)).to(device)

        x_next = mu_t + np.sqrt(eta) * sigma * z
        # print(t, "-> mu:", mu_w, "; sigma:", sigma, "; xt[1]:", x_next[0, 1])

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        else:
          mu[t] = mu_t.clone().cpu()
        # traj[0] = x_T_a + grad
        # break
        # if t == break_at_step:
        #     print("do break")
        #     # feat_acc = torch.zeros_like(feat_acc)
        #     # x_acc_a = torch.zeros_like(x_acc_a)
        #     # w_cnt = 0
        #     traj[0] = x_next.detach()
        #     break
    if ret_traj:
      return traj, mu
    else:
      return traj[0]

  # org
  def sample_gauss(
      self,
      x_T,
      sigma_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched

    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}
    mu = {}

    mt = int(var_sched.num_steps / 2)

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      print("ab gauss now")
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)

        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]

        # TODO: change it.
        # feat_t_ = self.featureNet(x_t)  # (B, N, F)

        # # NOTE:　mean
        # w = 1
        # feat_acc += feat_t_ * w
        # w_cnt += w
        # x_acc_a += x_t * w
        # feat_t = feat_acc / w_cnt
        # x_m = x_acc_a / w_cnt
        # print("mean")
        feat_t = feat_T
        x_m = x_T

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa,
                                 sigma_T)
        # e_theta = -1 * grad * sa_b / sa_b_i

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        mu_w = torch.nan_to_num(beta / (1 - alpha_bar), nan=0, posinf=0)
        sigma_bar = torch.sqrt((1 - alpha_bar) / alpha_bar)
        # mu_t = x_t + mu_w * grad / sigma_bar
        mu_t = x_t + mu_w * grad
        # x_next = mu_t
        x_next = mu_t / torch.sqrt(alpha) + torch.sqrt(beta) * z

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        else:
          mu[t] = mu_t.clone().cpu()
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj, mu
    else:
      return traj[0]

  # org
  def sample_sw(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
      slidWinSize=-1,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    if slidWinSize < 1:
      slidWinSize = var_sched.num_steps
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}
    mu = {}

    mt = int(var_sched.num_steps / 2)

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      F_n = feat_T.size(-1)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros(size=(slidWinSize, batch_size, num_points,
                                   F_n)).to(device)
      x_acc_a = torch.zeros(size=(slidWinSize, batch_size, num_points,
                                  point_dim)).to(device)
      w_cnt = 0
      mi = 0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        # TODO: rand it with the
        # z = (
        #     torch.randn(batch_size, num_points, 1).to(device)
        #     if t > 1
        #     else torch.zeros(batch_size, num_points, 1).to(device)
        # )
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        # alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        # sa = torch.sqrt(alpha)
        # b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        # NOTE:　mean
        # w = 1
        # feat_acc += feat_t_ * w
        # w_cnt += w
        # x_acc_a += x_t * w
        # feat_t = feat_acc / w_cnt
        # x_m = x_acc_a / w_cnt
        # print("mean")

        feat_acc[mi] = feat_t_
        x_acc_a[mi] = x_t
        w_cnt += 1
        if w_cnt > slidWinSize:
          w_cnt = slidWinSize
        feat_t = torch.sum(feat_acc, dim=0) / w_cnt
        x_m = torch.sum(x_acc_a, dim=0) / w_cnt

        x_T_m = x_acc_a[mi - w_cnt + 1]
        # print(mi, mi - w_cnt + 1, w_cnt, x_m.shape, x_m[0, 0])
        mi += 1
        if mi == slidWinSize:
          mi = 0

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T_m,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa)
        # e_theta = -1 * grad * sa_b / sa_b_i

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        mu_t = x_t + beta / (1 - alpha_bar) * grad
        # x_next = mu_t
        x_next = mu_t + sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm
        # print(t, "-> mu:", beta / (1 - alpha_bar), "; sigma:", sigma)

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        else:
          mu[t] = mu_t.clone().cpu()
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj, mu
    else:
      return traj[0]

  def sample_FF_FT(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    device = x_T.device

    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched

    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      # x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        # TODO: rand it with the
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        # alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        feat_t = feat_T
        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_T,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_T, Patchs_idx, self.kappa)
        # e_theta = -1 * grad * sa_b / sa_b_i

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        mu_t = x_t + beta / (1 - alpha_bar) * grad
        # x_next = mu_t
        x_next = mu_t + sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]

  def sample_FF_FT_wogf(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      var_sched=None,
      round=10,
  ):
    device = x_T.device

    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched

    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        # TODO: rand it with the
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        # feat_t = feat_T
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        w = 1
        feat_acc += feat_t_ * w
        w_cnt += w
        x_acc_a += x_t * w
        feat_t = feat_acc / w_cnt
        x_m = x_acc_a / w_cnt

        grad = self.SB_single_GridEstimate(
            x_m,
            x_T,
            x_t,
            feat_t,
            batch_size,
            num_points,
            point_dim,
            F_n,
            round,
        )
        # e_theta = -1 * grad * sa_b / sa_b_i

        # e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        # e_norm = torch.where(
        #     torch.isnan(e_norm), torch.full_like(e_norm, 0), e_norm
        # )
        # mu_t = x_t + beta / (1 - alpha_bar) * grad
        # # x_next = mu_t
        # x_next = mu_t + sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm
        mu_w = 1 - torch.nan_to_num(
            torch.sqrt((alpha - alpha_bar) / (1 - alpha_bar)),
            nan=0,
            posinf=0,
        )
        # print("eta=0: ", mu_w)
        x_next = x_t + mu_w * grad

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]

  def sample_langevin_dynamics(
      self,
      x_input,
      grad_compute_func,
      sample_knn,
      step_size,
      step_decay=0.95,
      num_steps=30,
      round=10,
      use_FF=True,
  ):
    device = x_input.device

    batch_size, num_points, point_dim = x_input.size()
    x_next = x_input.clone()
    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat = self.featureNet(x_input)  # (B, N, F)
      # print(feat.device, x_input.device)
      feat_acc = torch.zeros_like(feat)
      x_acc = torch.zeros_like(x_next)
      w_acc = 0
      _, _, F_n = feat.size()
      # traj = [x_next.clone().cpu()]
      # for t in range(var_sched.num_steps, 0, -1):
      for step in tqdm(range(num_steps), desc="Sampling"):
        if use_FF:
          feat_acc += self.featureNet(x_next)
          x_acc += x_next
          w_acc += 1
          feat_t = feat_acc / w_acc
          x_m = x_acc / w_acc
          grad_pred, Patchs_idx = self.SB_GridEstimate(
              x_input,
              x_m,
              x_next,
              feat_t,
              batch_size,
              num_points,
              sample_knn,
              point_dim,
              F_n,
              round,
          )
        else:
          # print(x_next.shape)
          # print(batch_size, num_points, sample_knn, point_dim, F_n)
          grad_pred, Patchs_idx = self.SB_GridEstimate(
              x_input,
              x_input,
              x_next,
              feat,
              batch_size,
              num_points,
              sample_knn,
              point_dim,
              F_n,
              round,
          )

        grad = grad_compute_func(grad_pred, x_next, x_input, Patchs_idx,
                                 self.kappa)
        s = step_size * (step_decay**step)
        x_next += s * grad

        # traj.append(x_next.detach().clone().cpu())

    return x_next

  def sample_end2end(
      self,
      x_input,
      grad_compute_func,
      sample_knn,
      num_steps=2,
      round=10,
      use_FF=True,
  ):
    device = x_input.device

    batch_size, num_points, point_dim = x_input.size()
    x_next = x_input.clone()
    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat = self.featureNet(x_input)  # (B, N, F)
      # print(feat.device, x_input.device)
      feat_acc = torch.zeros_like(feat)
      x_acc = torch.zeros_like(x_next)
      w_acc = 0
      _, _, F_n = feat.size()
      # traj = [x_next.clone().cpu()]
      # for t in range(var_sched.num_steps, 0, -1):
      for step in range(num_steps):
        if use_FF:
          feat_acc += self.featureNet(x_next)
          x_acc += x_next
          w_acc += 1
          feat_t = feat_acc / w_acc
          x_m = x_acc / w_acc
          grad_pred, Patchs_idx = self.SB_GridEstimate(
              x_input,
              x_m,
              x_next,
              feat_t,
              batch_size,
              num_points,
              sample_knn,
              point_dim,
              F_n,
              round,
          )
        else:
          # print(x_next.shape)
          # print(batch_size, num_points, sample_knn, point_dim, F_n)
          grad_pred, Patchs_idx = self.SB_GridEstimate(
              x_input,
              x_input,
              x_next,
              feat,
              batch_size,
              num_points,
              sample_knn,
              point_dim,
              F_n,
              round,
          )

        grad = grad_compute_func(grad_pred, x_next, x_input, Patchs_idx,
                                 self.kappa)

        x_next += grad

        # traj.append(x_next.detach().clone().cpu())

    return x_next

  def sample_FF_Ft(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    device = x_T.device

    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched

    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      # x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        # TODO: rand it with the
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        # alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.
        feat_t = self.featureNet(x_t)
        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_t,
            x_t,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_t, Patchs_idx, self.kappa)
        # e_theta = -1 * grad * sa_b / sa_b_i

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        x_next = (x_t + beta / (1 - alpha_bar) * grad +
                  sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm)

        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]

  def sample_Li(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        sa = torch.sqrt(alpha)
        b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]

        # TODO: change it.
        # feat_t_ = self.featureNet(x_t)  # (B, N, F)

        # w = sa_b_i / sa_b / noiseL
        # feat_acc += feat_t_ * w
        # w_cnt += w
        # x_acc_a += x_t * w
        # feat_t = feat_acc / w_cnt
        # x_m = x_acc_a / w_cnt\
        feat_t = feat_T

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_T,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_T, Patchs_idx, self.kappa)

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        # x_next = (
        #     x_t
        #     + beta / (1 - alpha_bar) * grad
        #     + sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm
        # )
        # print(t, beta / (1 - alpha_bar))
        # x_next = x_t + beta / (1 - alpha_bar) * grad
        e_theta = -grad
        x_next = (x_t - b / sa_b_i * e_theta
                 ) / sa + sigma * sa_b / sa * z[:, :,
                                                0].unsqueeze(dim=-1) * e_norm

        # alpha_bar_next = var_sched.alpha_bars[t - 1]
        # c_a = 1.0 / torch.sqrt(alpha_bar_next)
        # x_next = x_next * c_a
        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]

  def sample_Lg(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        sa = torch.sqrt(alpha)
        b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]

        # TODO: change it.
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        w = sa_b_i / sa_b / noiseL
        feat_acc += feat_t_ * w
        w_cnt += w
        x_acc_a += x_t * w
        feat_t = feat_acc / w_cnt
        x_m = x_acc_a / w_cnt

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa)

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        x_next = x_t + beta / (1 - alpha_bar) * grad + sigma * z
        # print(t, beta / (1 - alpha_bar))
        # x_next = x_t + beta / (1 - alpha_bar) * grad
        # e_theta = -grad
        # x_next = (x_t - b / sa_b_i * e_theta) / sa + sigma * sa_b / sa * z[
        #     :, :, 0
        # ].unsqueeze(dim=-1) * e_norm

        # alpha_bar_next = var_sched.alpha_bars[t - 1]
        # c_a = 1.0 / torch.sqrt(alpha_bar_next)
        # x_next = x_next * c_a
        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]

  def sample_LiLg(
      self,
      x_T,
      grad_compute_func,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
  ):
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()
    if var_sched == None:
      var_sched = self.var_sched
    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    noiseL = var_sched.vs_sigma_acc()
    traj = {var_sched.num_steps: x_T.clone()}

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      feat_acc = torch.zeros_like(feat_T)
      # x_acc_a = torch.zeros_like(x_T)
      w_cnt = 0
      F_n = feat_T.size(-1)
      a_step = 0.03
      a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
        z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        alpha = var_sched.alphas[t]
        alpha_bar = var_sched.alpha_bars[t]
        beta = var_sched.betas[t]
        sigma = var_sched.get_sigmas(t, flexibility)
        sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        sa = torch.sqrt(alpha)
        b = beta
        sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]

        # TODO: change it.
        feat_t_ = self.featureNet(x_t)  # (B, N, F)

        w = sa_b_i / sa_b / noiseL
        feat_acc += feat_t_ * w
        w_cnt += w
        x_acc_a += x_t * w
        feat_t = feat_acc / w_cnt
        x_m = x_acc_a / w_cnt

        grad_pred, Patchs_idx = self.SB_GridEstimate(
            x_T,
            x_m,
            x_t,
            feat_t,
            batch_size,
            num_points,
            sample_knn,
            point_dim,
            F_n,
            round,
        )

        grad = grad_compute_func(grad_pred, x_t, x_m, Patchs_idx, self.kappa)

        e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
        e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
                             e_norm)
        # x_next = (
        #     x_t
        #     + beta / (1 - alpha_bar) * grad
        #     + sigma * z[:, :, 0].unsqueeze(dim=-1) * e_norm
        # )
        # print(t, beta / (1 - alpha_bar))
        # x_next = x_t + beta / (1 - alpha_bar) * grad
        e_theta = -grad
        x_next = (x_t - b / sa_b_i * e_theta) / sa + sigma * sa_b / sa * z

        # alpha_bar_next = var_sched.alpha_bars[t - 1]
        # c_a = 1.0 / torch.sqrt(alpha_bar_next)
        # x_next = x_next * c_a
        traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
        traj[t] = traj[t].clone().cpu()  # Move previous output to CPU memory.
        if not ret_traj:
          del traj[t]
        # traj[0] = x_T_a + grad
        # break
    if ret_traj:
      return traj
    else:
      return traj[0]
