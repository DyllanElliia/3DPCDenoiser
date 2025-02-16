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
      from .encoders.dgcnn0323 import FeatureExtraction

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
    if args.tag == "FF0528QK":
      print("FF0528QK")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionQKBlock
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionQKBlock(c_dim=self.featureOut_channels)
    elif args.tag == "FF0528QKL":
      print("FF0528QKL")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionQKBlock_2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionQKBlock_2(c_dim=self.featureOut_channels)
    elif args.tag == "FF0528LPL":
      print("FF0528LPL")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_2(c_dim=self.featureOut_channels)
    elif args.tag == "FF0615LPL":
      print("FF0615LPL")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0615
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0615(c_dim=self.featureOut_channels)
    elif args.tag == "FF0618LPL":
      print("FF0618LPL")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0618
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0618(c_dim=self.featureOut_channels)
    elif args.tag == "FF0528LP":
      print("FF0528LP")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionLerpBlock
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock(c_dim=self.featureOut_channels)
    elif args.tag == "FF0528AT":
      print("FF0528AT")
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionAttenBlock
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionAttenBlock(c_dim=self.featureOut_channels)
    else:
      from .decoders.score_FF0528 import ScoreNet
      from .decoders.fusion import FusionBlock
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionBlock(c_dim=self.featureOut_channels)
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
    # self.train_sigma=self.var_sched.

  def get_loss(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    batch_size, point_num, point_dim = x_0.size()

    t = self.var_sched.uniform_sample_t(batch_size)
    t = self.randomFunc_Tt(t, self.var_sched)

    alpha_bar_t = self.var_sched.alpha_bars[t]

    sa_b_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)
    sa_b_i_t = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)

    t = torch.tensor(t).to(x_0.device)
    tT = rearrange(t, "B -> B () ()") / self.var_sched.num_steps

    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    e_rand = 0.03 * torch.randn_like(x_0)  # (B, N, d)
    # NOTE: evaluate GuassianNoise
    x_t_a = x_0 + tT * e_rand

    feat = self.featureNet(x_t_a)  # (B, N, F)

    if self.use_patch_ratio:
      xt_p_idx = get_random_indices(int(point_num / self.patch_ratio),
                                    self.num_train_points)
    else:
      xt_p_idx = get_random_indices(point_num, self.num_train_points)
    feat = feat[:, xt_p_idx, :]  # (B, n, F)
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]  # (B,n,3)
    feat = self.fusion(feat, feat, x_t_xt_p_index, x_t_xt_p_index,
                       torch.ones(batch_size).to(x_0.device))
    F_n = feat.size(-1)

    # Local Patch construction
    _, _, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)

    # use Score-Based compute grad
    grad_pred = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
              point_dim)  # (B, n, K, 3)

    # Local Patch construction for Clear Point
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
    # compute target vectors
    gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    gt_target = -1 * gt_offset

    # sig2 = ((self.alpha_mix +
    #          (1 - self.alpha_mix) * sa_b_t / sa_b_i_t)**2).view(-1, 1, 1, 1)
    # loss = (sig2 * (grad_pred - gt_target)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma)
    loss = ((grad_pred - gt_target)**
            2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    # phase 2
    t_s = torch.randn(batch_size).to(x_0.device)
    tt = rearrange(t_s, "B -> B () ()")

    _, _, x_0_nn = pytorch3d.ops.knn_points(x_t_a, x_0, K=1,
                                            return_nn=True)  # (B, N, 1, 3)
    # print(x_0_nn.shape)
    x_0_nn = rearrange(x_0_nn, "B N 1 d -> B N d")

    e_rand = x_0_nn - x_t_a
    # NOTE: evaluate GuassianNoise
    x_t_a = x_t_a + (1 - tt) * e_rand  # (B, N, d)

    feat_t = self.featureNet(x_t_a)  # (B, N, F)
    feat_t = feat_t[:, xt_p_idx, :]  # (B, n, F)
    x_T_xt_p_index = x_t_xt_p_index.clone()
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]
    feat = self.fusion(feat, feat_t, x_T_xt_p_index, x_t_xt_p_index, t_s)
    F_n = feat.size(-1)

    # Local Patch construction
    _, _, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)

    # use Score-Based compute grad
    grad_pred = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
              point_dim)  # (B, n, K, 3)

    # Local Patch construction for Clear Point
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
    # compute target vectors
    gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    gt_target = -1 * gt_offset

    loss += ((grad_pred - gt_target)**
             2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    return loss

  def SB_GridEstimate(self, x_T_a, x_m_a, x_t_a, feat, B, N, K, d, F, round=10):
    index = [list(range(0 + i, N, round)) for i in range(round)]
    grad_pred = torch.zeros((B, N, K, d), device=x_T_a.device)
    Patchs_idx = torch.zeros((B, N, K), dtype=torch.int64, device=x_T_a.device)

    for x_r_i in index:
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
          c=rearrange(feat[:, x_r_i, :], "b n f -> (b n) f"),  # (B*Ni, F)
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
        c=rearrange(feat, "b n f -> (b n) f")  # (B*N, 1, 3)  # (B*N, F)
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
      self.fusion.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_T = self.fusion(feat_T, feat_T, x_T, x_T,
                           torch.tensor([1.0]).to(x_T.device))
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
      self.fusion.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_T = self.fusion(feat_T, feat_T, x_T, x_T,
                           torch.tensor([1.0]).to(x_T.device))
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
          feat_t = self.featureNet(x_next)
          feat_t = self.fusion(
              feat, feat_t, x_input, x_next,
              torch.tensor([step / num_steps]).to(x_input.device))
          grad_pred, Patchs_idx = self.SB_GridEstimate(
              x_input,
              x_next,
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

        if feature_fusion:
          # w = 1
          # feat_acc += feat_t_ * w
          # w_cnt += w
          # x_acc_a += x_t * w
          # feat_t = feat_acc / w_cnt
          # x_m = x_acc_a / w_cnt  # (B, N, 3)
          feat_t = self.fusion(
              feat_T, feat_t_, x_T, x_t,
              torch.tensor([t / var_sched.num_steps]).to(device))
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
  # TODO: update
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

    if ret_traj:
      return traj, mu
    else:
      return traj[0]
