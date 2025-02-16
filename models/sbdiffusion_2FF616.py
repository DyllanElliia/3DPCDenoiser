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


def grad_compute_neural(grad_pred,
                        x_t,
                        grad_weight,
                        Patchs_idx,
                        ret_mask=False):
  """
    Args:
        grad_pred:  Input grad prediction, (B, N, K, 3)
        x_t: Point clouds, (B, N, 3)
        grad_weight: neural weight of grad, (B, N, K, 1)
        Patchs_idx: Point Patch indices, (B, N, K)
    """
  # NOTE: single score

  pdf = grad_weight  # (B, N, K, 1)

  g_w = (grad_pred * pdf).reshape(x_t.size(0), -1, x_t.size(2))  # (B, N*K, 3)
  nn_idx = Patchs_idx.reshape(x_t.size(0), -1)  # (B, N*K,)
  nn_idx = nn_idx.unsqueeze(-1).expand_as(g_w)  # (B, N*K, 1) -> (B, N*K, 3)
  acc_g = torch.zeros_like(x_t)
  acc_g.scatter_add_(dim=1, index=nn_idx, src=g_w)

  pdf = pdf.expand_as(grad_pred).reshape(x_t.size(0), -1,
                                         x_t.size(2))  # (B, N*K, 3)
  acc_pdf = torch.zeros_like(x_t)
  acc_pdf.scatter_add_(dim=1, index=nn_idx, src=pdf)
  if ret_mask:
    mask = torch.ones_like(acc_pdf)
    mask[acc_pdf < 1e-10] = 0
    acc_pdf[acc_pdf < 1e-10] = 1
    acc_grad = acc_g / acc_pdf

    return acc_grad, mask
  else:
    acc_pdf[acc_pdf < 1e-10] = 1
    acc_grad = acc_g / acc_pdf

    return acc_grad


class Exp(Module):

  def forward(self, x):
    return torch.exp(x)


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
    if hasattr(args, "dsm_sigma2"):
      self.dsm_sigma2 = args.dsm_sigma2
    else:
      self.dsm_sigma2 = self.dsm_sigma
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
    if hasattr(args, "use_patch"):
      self.use_patch = args.use_patch
      print("use_patch<-{}".format(args.use_patch))
    else:
      self.use_patch = False
      print("use_patch<-{}".format(args.use_patch))
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
    elif args.tag == "FFF0703LPL2FE2":
      from .encoders.EdgeCNN0716 import FeatureExtraction
      print("FFF0703LPL2FE2")

      self.featureNet = FeatureExtraction()
    elif args.tag == "FFF0703LPL2OFE2":
      from .encoders.denseDgcnn2 import FeatureExtraction
      print("FFF0703LPL2OFE2")
      self.featureNet = FeatureExtraction(
          conv_knn=self.edge_conv_knn,
          conv_num_fc_layers=self.conv_num_fc_layers,
          conv_growth_rate=self.conv_growth_rate,
          num_convs=self.num_convs,
          conv_channels=self.conv_channels,
          mlp_feat=self.mlp_feat,
          cat_feat=self.cat_feat,
      )
    elif args.tag == "FFF0802LPL2AnotherSoftmax":
      from .encoders.EdgeCNN0718 import FeatureExtraction
      self.featureNet = FeatureExtraction(embedding_dim=512)
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
    elif args.tag == "FFF0615LPL":
      print("FFF0622LPL")
      from .decoders.score_FF0616 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0615
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0615(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0702LPL":
      print("FFF0702LPL")
      from .decoders.score_FF0319v2 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0615
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0615(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0703LPL":
      print("FFF0703LPL")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0703(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0703LPL2":
      print("FFF0703LPL2")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0728LPL2":
      print("FFF0728LPL2")
      from .decoders.score_FF0319v3_3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2":
      print("FFF0802LPL2")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          # fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2softmax":
      print("FFF0802LPL2softmax")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2AnotherSoftmax":
      print("FFF0802LPL2AnotherSoftmax")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2FmlpSoftmax":
      print("FFF0802LPL2FmlpSoftmax")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703_mo
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703_mo(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2F2lmlpSoftmax":
      print("FFF0802LPL2F2lmlpSoftmax")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703_mo2l
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703_mo2l(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0802LPL2F2lmlpSoftmaxV2":
      print("FFF0802LPL2F2lmlpSoftmaxV2")
      from .decoders.score_FF0319v4 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703_mo2lv2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          fusion_model=Exp(),
      )
      self.fusion = FusionLerpBlock_0703_mo2lv2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0703LPL2FE2":
      print("FFF0703LPL2FE2")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          decoder_h=args.score_net_decoder_h,
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0703LPL2OFE2":
      print("FFF0703LPL2OFE2")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0703v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
          decoder_h=args.score_net_decoder_h,
      )
      self.fusion = FusionLerpBlock_0703v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0705LPL":
      print("FFF0705LPL")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0705v2
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0705v2(c_dim=self.featureOut_channels)
    elif args.tag == "FFF0705LPL3":
      print("FFF0705LPL3")
      from .decoders.score_FF0319v3 import ScoreNet
      from .decoders.fusion import FusionLerpBlock_0705v3
      self.scoreNet = ScoreNet(
          z_dim=self.featureOut_channels,
          dim=3,
          out_dim=3,
          hidden_size=args.score_net_hidden_dim,
          num_blocks=args.score_net_num_blocks,
      )
      self.fusion = FusionLerpBlock_0705v3(c_dim=self.featureOut_channels)
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
    if hasattr(args, "normalize_patch"):
      self.normalize_patch = args.normalize_patch
    else:
      self.normalize_patch = True

    if hasattr(args, "train_phase2_mode_switch"):
      self.train_phase2_mode_switch = args.train_phase2_mode_switch
    else:
      self.train_phase2_mode_switch = -1
    print("train_phase2_mode_switch:", self.train_phase2_mode_switch)
    # variables (for training)
    self.var_sched = var_sched
    # self.train_sigma=self.var_sched.
    self.train_noise = 0.035  # default 0.035
    self.tag = args.tag
    self.ctx_noiseL = 0.03
    self.ctx_num_steps = 3
    self.denoise_mode = 3
    if self.denoise_mode == 0 or self.denoise_mode == 1:
      print("use get_loss_old")
      self.get_loss = self.get_loss_old
    elif self.denoise_mode == 2:
      print("use get_loss_v2")
      self.get_loss = self.get_loss_v2
    elif self.denoise_mode == 3:
      print("use get_loss_v3")
      self.get_loss = self.get_loss_v3
    elif self.denoise_mode == 4:
      print("use get_loss_v4")
      self.get_loss = self.get_loss_v4
    print("var beta: 0={} T={}".format(self.var_sched.betas[0],
                                       self.var_sched.betas[-1]))

  def load_feat_net(self, feat_ckpt):
    self.featureNet.load_state_dict(feat_ckpt)

  def set_feat_net_requires_grad(self, requires_grad):
    self.featureNet._requires_grad_ = requires_grad
    self.featureNet.requires_grad_(requires_grad)

  def get_patch_distance(self, patch, patch_center, normalize=True):
    dis = torch.norm(patch - patch_center, dim=-1, keepdim=True)
    if normalize:
      dis = dis / dis.max()
    return dis

  def get_dist_weight_gauss(self, dist_weight):
    dist_weight_sq = dist_weight**2  # (B,n,1)
    max_dist = dist_weight_sq[:, -1, :].unsqueeze(1)  # (B,1,1)
    dist_weight_sq = 9 * dist_weight_sq / max_dist  # (B,n,1)
    dist_weight_sq = torch.exp(-dist_weight_sq)  # (B,n,1)
    dist_weight_sq_sum = dist_weight_sq.sum(dim=1, keepdim=True)  # (B,1,1)
    return dist_weight_sq / dist_weight_sq_sum  # (B,n,1)

  def get_loss_old(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    batch_size, point_num, point_dim = x_0.size()
    loss = 0
    t = self.var_sched.uniform_sample_t(batch_size)
    t = self.randomFunc_Tt(t, self.var_sched)

    alpha_bar_t = self.var_sched.alpha_bars[t]

    sa_b_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)
    sa_b_i_t = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)

    t = torch.tensor(t).to(x_0.device)
    tT = rearrange(t, "B -> B () ()") / self.var_sched.num_steps

    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    e_rand = self.train_noise * torch.randn_like(x_0)  # (B, N, d)
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

    # TODO: remove fusion in t=1
    # feat = self.fusion(feat, feat, x_t_xt_p_index, x_t_xt_p_index,
    #                    torch.ones(batch_size).to(x_0.device))
    F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    if self.normalize_patch:
      Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)
    else:
      Patchs_center = Patchs

    # use Score-Based compute grad
    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    grad_fusion, mask = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
                                            True)  # (B,N,3)
    # Local Patch construction for Clear Point
    # _, _, clean_Patchs = pytorch3d.ops.knn_points(
    #     Patchs.view(-1, self.frame_knn, point_dim),  # (B*n, K, 3)
    #     x_0.unsqueeze(1).repeat(1, len(xt_p_idx), 1,
    #                             1).view(-1, point_num,
    #                                     point_dim),  # (B*n, M, 3)
    #     K=self.num_clean_nbs,
    #     return_nn=True,
    # )  # (B*n, K, C, 3)
    # clean_Patchs = clean_Patchs.view(batch_size, len(xt_p_idx), self.frame_knn,
    #                                  self.num_clean_nbs,
    #                                  point_dim)  # (B, n, K, C, 3)
    # # # compute target vectors
    # gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    # gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    # gt_target = -1 * gt_offset
    # loss += ((grad_pred - gt_target)**
    #          2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs  # x_0 to x_t
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # x_t to x_0

    if self.use_patch:
      # # dis_weight = self.get_patch_distance(x_t_a, x_0[:, :1, :])**2  # (B,n,1)
      # # dis_weight = torch.abs(1. - dis_weight)
      # dis_weight = self.get_patch_distance(x_t_a,
      #                                      x_0[:, :1, :],
      #                                      normalize=False)  # (B,n,1)
      # dis_weight = self.get_dist_weight_gauss(dis_weight)  # (B,n,1)
      dis_weight = 1.0
    else:
      dis_weight = 1.0
    # print(dis_weight.shape)
    # print("mash mean:", mask.mean())
    # print("weight mean", grad_weight.mean())
    s2 = (self.alpha_mix + (1 - self.alpha_mix) /
          (self.train_noise * tT))**2  # (B,1,1)
    s2 = torch.clamp(s2, max=1000.0)
    # print(tT, s2)
    # loss += (mask * s2 * dis_weight *
    #          (grad_fusion - gt_offset_fusion)**2.0).sum(
    #              dim=-1).mean() * (1.0 / self.dsm_sigma)
    loss += ((mask * s2 * dis_weight *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma)).mean(dim=-1)
    # phase 2
    t_s = torch.randn(batch_size).to(x_0.device)
    tt = rearrange(t_s, "B -> B () ()")

    if self.train_phase2_mode_switch == 1:
      e_rand = ((1 - mask) * gt_offset_fusion + mask * grad_fusion)
    elif self.train_phase2_mode_switch == 2:
      e_rand = gt_offset_fusion
    else:
      _, _, x_0_nn = pytorch3d.ops.knn_points(x_t_a, x_0, K=1,
                                              return_nn=True)  # (B, N, 1, 3)

      x_0_nn = rearrange(x_0_nn, "B N 1 d -> B N d")
      e_rand = x_0_nn - x_t_a  # x_t to x_0

    # NOTE: evaluate GuassianNoise
    x_t_a = x_t_a + (1 - tt) * e_rand  # (B, N, d)

    feat_t = self.featureNet(x_t_a)  # (B, N, F)
    feat_t = feat_t[:, xt_p_idx, :]  # (B, n, F)
    x_T_xt_p_index = x_t_xt_p_index.clone()
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]
    feat = self.fusion(feat, feat_t, x_T_xt_p_index, x_t_xt_p_index, t_s)
    F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    if self.normalize_patch:
      Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)
    else:
      Patchs_center = Patchs

    # use Score-Based compute grad
    # grad_pred = self.scoreNet(
    #     x=Patchs_center.view(-1, self.frame_knn, point_dim),
    #     c=rearrange(feat, "b n f -> (b n) f"),
    # ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
    #           point_dim)  # (B, n, K, 3)

    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    grad_fusion, mask = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
                                            True)  # (B,N,3)

    # _, _, clean_Patchs = pytorch3d.ops.knn_points(
    #     Patchs.view(-1, self.frame_knn, point_dim),  # (B*n, K, 3)
    #     x_0.unsqueeze(1).repeat(1, len(xt_p_idx), 1,
    #                             1).view(-1, point_num,
    #                                     point_dim),  # (B*n, M, 3)
    #     K=self.num_clean_nbs,
    #     return_nn=True,
    # )  # (B*n, K, C, 3)
    # clean_Patchs = clean_Patchs.view(batch_size, len(xt_p_idx), self.frame_knn,
    #                                  self.num_clean_nbs,
    #                                  point_dim)  # (B, n, K, C, 3)
    # # compute target vectors
    # gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    # gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    # gt_target = -1 * gt_offset

    # loss += ((grad_pred - gt_target)**
    #          2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # (B, N, 3)

    if self.use_patch:
      # # dis_weight = self.get_patch_distance(x_t_a, x_0[:, :1, :])**2  # (B,n,1)
      # # dis_weight = torch.abs(1. - dis_weight)
      # dis_weight = self.get_patch_distance(x_t_a,
      #                                      x_0[:, :1, :],
      #                                      normalize=False)  # (B,n,1)
      # dis_weight = self.get_dist_weight_gauss(dis_weight)  # (B,n,1)
      dis_weight = 1.0
    else:
      dis_weight = 1.0
    # sig2 = ((self.alpha_mix +
    #          (1 - self.alpha_mix) * sa_b_t / sa_b_i_t)**2).view(-1, 1, 1, 1)
    # loss = (sig2 * (grad_pred - gt_target)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma)
    # loss = ((grad_pred - gt_target)**
    #         2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)
    # s2 = (self.alpha_mix + (1 - self.alpha_mix) /
    #       (self.train_noise * tt * tT))**2  # (B,1,1)
    # s2 = torch.clamp(s2, max=100.0)
    # loss += (mask * dis_weight * (grad_fusion - gt_offset_fusion)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma2)
    # loss += (mask * dis_weight * (grad_fusion - gt_offset_fusion)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma2)

    # 1
    loss += ((mask * dis_weight *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma2)).mean(dim=-1)

    # 2
    # s2 = (self.alpha_mix + (1 - self.alpha_mix) /
    #       (self.train_noise * tt * tT))**2  # (B,1,1)
    # s2 = torch.clamp(s2, max=5.0)
    # loss += ((mask * s2 * dis_weight *
    #           (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
    #          (1.0 / self.dsm_sigma2)).mean(dim=-1)

    return loss

  def get_loss_v2(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    batch_size, point_num, point_dim = x_0.size()
    loss = 0
    t = self.var_sched.uniform_sample_t(batch_size)
    t = self.randomFunc_Tt(t, self.var_sched)

    alpha_bar_t = self.var_sched.alpha_bars[t]

    sa_b_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)
    sa_b_i_t = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1)  # (B, 1, 1)

    t = torch.tensor(t).to(x_0.device)
    tT = rearrange(t, "B -> B () ()") / self.var_sched.num_steps

    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    e_rand = self.train_noise * torch.randn_like(x_0)  # (B, N, d)
    # NOTE: evaluate GuassianNoise
    x_t_a = x_0 + tT * e_rand

    feat = self.featureNet(x_t_a)  # (B, N, F)

    # if self.use_patch_ratio:
    #   xt_p_idx = get_random_indices(int(point_num / self.patch_ratio),
    #                                 self.num_train_points)
    # else:
    #   xt_p_idx = get_random_indices(point_num, self.num_train_points)

    # get top point_num index
    xt_p_idx = torch.arange(self.num_train_points).to(x_0.device)
    # print(x_t_a.shape, xt_p_idx.shape)

    feat = feat[:, xt_p_idx, :]  # (B, n, F)
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]  # (B,n,3)

    # TODO: remove fusion in t=1
    feat = self.fusion(feat, feat, x_t_xt_p_index, x_t_xt_p_index,
                       torch.ones(batch_size).to(x_0.device))
    F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    if self.normalize_patch:
      Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)
    else:
      Patchs_center = Patchs

    # use Score-Based compute grad
    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    grad_fusion, mask = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
                                            True)  # (B,N,3)
    # Local Patch construction for Clear Point
    # _, _, clean_Patchs = pytorch3d.ops.knn_points(
    #     Patchs.view(-1, self.frame_knn, point_dim),  # (B*n, K, 3)
    #     x_0.unsqueeze(1).repeat(1, len(xt_p_idx), 1,
    #                             1).view(-1, point_num,
    #                                     point_dim),  # (B*n, M, 3)
    #     K=self.num_clean_nbs,
    #     return_nn=True,
    # )  # (B*n, K, C, 3)
    # clean_Patchs = clean_Patchs.view(batch_size, len(xt_p_idx), self.frame_knn,
    #                                  self.num_clean_nbs,
    #                                  point_dim)  # (B, n, K, C, 3)
    # # # compute target vectors
    # gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    # gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    # gt_target = -1 * gt_offset
    # loss += ((grad_pred - gt_target)**
    #          2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs  # x_0 to x_t
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # x_t to x_0

    if self.use_patch:
      # # dis_weight = self.get_patch_distance(x_t_a, x_0[:, :1, :])**2  # (B,n,1)
      # # dis_weight = torch.abs(1. - dis_weight)
      # dis_weight = self.get_patch_distance(x_t_a,
      #                                      x_0[:, :1, :],
      #                                      normalize=False)  # (B,n,1)
      # dis_weight = self.get_dist_weight_gauss(dis_weight)  # (B,n,1)
      dis_weight = 1.0
    else:
      dis_weight = 1.0
    # print(dis_weight.shape)
    # print("mash mean:", mask.mean())
    # print("weight mean", grad_weight.mean())
    s2 = (self.alpha_mix + (1 - self.alpha_mix) /
          (self.train_noise * tT))**2  # (B,1,1)
    s2 = torch.clamp(s2, max=1000.0)
    # print(tT, s2)
    # loss += (mask * s2 * dis_weight *
    #          (grad_fusion - gt_offset_fusion)**2.0).sum(
    #              dim=-1).mean() * (1.0 / self.dsm_sigma)
    loss += ((mask * s2 * dis_weight *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma)).mean(dim=-1)
    # phase 2
    t_s = torch.randn(batch_size).to(x_0.device)
    tt = rearrange(t_s, "B -> B () ()")

    if self.train_phase2_mode_switch == 1:
      e_rand = ((1 - mask) * gt_offset_fusion + mask * grad_fusion)
    elif self.train_phase2_mode_switch == 2:
      e_rand = gt_offset_fusion
    else:
      _, _, x_0_nn = pytorch3d.ops.knn_points(x_t_a, x_0, K=1,
                                              return_nn=True)  # (B, N, 1, 3)

      x_0_nn = rearrange(x_0_nn, "B N 1 d -> B N d")
      e_rand = x_0_nn - x_t_a  # x_t to x_0

    # NOTE: evaluate GuassianNoise
    x_t_a = x_t_a + (1 - tt) * e_rand  # (B, N, d)

    feat_t = self.featureNet(x_t_a)  # (B, N, F)
    feat_t = feat_t[:, xt_p_idx, :]  # (B, n, F)
    x_T_xt_p_index = x_t_xt_p_index.clone()
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]
    feat = self.fusion(feat, feat_t, x_T_xt_p_index, x_t_xt_p_index, t_s)
    F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)
    if self.normalize_patch:
      Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)
    else:
      Patchs_center = Patchs

    # use Score-Based compute grad
    # grad_pred = self.scoreNet(
    #     x=Patchs_center.view(-1, self.frame_knn, point_dim),
    #     c=rearrange(feat, "b n f -> (b n) f"),
    # ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
    #           point_dim)  # (B, n, K, 3)

    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    grad_fusion, mask = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
                                            True)  # (B,N,3)

    # _, _, clean_Patchs = pytorch3d.ops.knn_points(
    #     Patchs.view(-1, self.frame_knn, point_dim),  # (B*n, K, 3)
    #     x_0.unsqueeze(1).repeat(1, len(xt_p_idx), 1,
    #                             1).view(-1, point_num,
    #                                     point_dim),  # (B*n, M, 3)
    #     K=self.num_clean_nbs,
    #     return_nn=True,
    # )  # (B*n, K, C, 3)
    # clean_Patchs = clean_Patchs.view(batch_size, len(xt_p_idx), self.frame_knn,
    #                                  self.num_clean_nbs,
    #                                  point_dim)  # (B, n, K, C, 3)
    # # compute target vectors
    # gt_offset = Patchs.unsqueeze(3) - clean_Patchs  # (B, n, K, C, 3)
    # gt_offset = gt_offset.mean(3)  # (B, n, K, 3)
    # # gt_offset = Patchs - clean_Patchs[:, :, :, 0, :]  # (B, n, K, 3)
    # gt_target = -1 * gt_offset

    # loss += ((grad_pred - gt_target)**
    #          2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # (B, N, 3)

    if self.use_patch:
      # # dis_weight = self.get_patch_distance(x_t_a, x_0[:, :1, :])**2  # (B,n,1)
      # # dis_weight = torch.abs(1. - dis_weight)
      # dis_weight = self.get_patch_distance(x_t_a,
      #                                      x_0[:, :1, :],
      #                                      normalize=False)  # (B,n,1)
      # dis_weight = self.get_dist_weight_gauss(dis_weight)  # (B,n,1)
      dis_weight = 1.0
    else:
      dis_weight = 1.0
    # sig2 = ((self.alpha_mix +
    #          (1 - self.alpha_mix) * sa_b_t / sa_b_i_t)**2).view(-1, 1, 1, 1)
    # loss = (sig2 * (grad_pred - gt_target)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma)
    # loss = ((grad_pred - gt_target)**
    #         2.0).sum(dim=-1).mean() * (1.0 / self.dsm_sigma)
    # s2 = (self.alpha_mix + (1 - self.alpha_mix) /
    #       (self.train_noise * tt * tT))**2  # (B,1,1)
    # s2 = torch.clamp(s2, max=100.0)
    # loss += (mask * dis_weight * (grad_fusion - gt_offset_fusion)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma2)
    # loss += (mask * dis_weight * (grad_fusion - gt_offset_fusion)**2.0).sum(
    #     dim=-1).mean() * (1.0 / self.dsm_sigma2)

    # 1
    loss += ((mask * dis_weight *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma2)).mean(dim=-1)

    # 2
    # s2 = (self.alpha_mix + (1 - self.alpha_mix) /
    #       (self.train_noise * tt * tT))**2  # (B,1,1)
    # s2 = torch.clamp(s2, max=5.0)
    # loss += ((mask * s2 * dis_weight *
    #           (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
    #          (1.0 / self.dsm_sigma2)).mean(dim=-1)

    return loss

  def get_loss_v3(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    # print("var beta: 0={} T={}".format(self.var_sched.betas[0],
    #                                    self.var_sched.betas[-1]))
    # print("Max sigma: {}".format(((1 - self.var_sched.alpha_bars[-1]) /
    #                               self.var_sched.alpha_bars[-1])**0.5))

    batch_size, point_num, point_dim = x_0.size()
    loss = 0
    t = self.var_sched.uniform_sample_t(batch_size)
    t = self.randomFunc_Tt(t, self.var_sched)

    t = torch.tensor(t).to(x_0.device)
    alpha_bar_t = self.var_sched.alpha_bars[t]
    tT = rearrange(t, "B -> B () ()") / self.var_sched.num_steps
    sigma_t = torch.sqrt((1 - alpha_bar_t) / alpha_bar_t).view(-1, 1, 1)
    # print("t", t)
    # print("sigma_t", sigma_t)

    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    # e_rand = self.train_noise * torch.randn_like(x_0)  # (B, N, d)
    # NOTE: evaluate GuassianNoise
    # x_t_a = x_0 + tT * e_rand
    e_rand = torch.randn_like(x_0)  # (B, N, d)
    x_t_a = x_0 + sigma_t * e_rand

    feat_T = self.featureNet(x_t_a)  # (B, N, F)

    # get top point_num index
    xt_p_idx = torch.arange(self.num_train_points).to(x_0.device)
    mask = torch.zeros_like(x_0).to(x_0.device)
    mask[:, xt_p_idx, :] = 1
    # print(x_t_a.shape, xt_p_idx.shape)

    # feat = feat[:, xt_p_idx, :]  # (B, n, F)
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]  # (B,n,3)

    # TODO: remove fusion in t=1
    # feat = self.fusion(feat, feat, x_t_xt_p_index, x_t_xt_p_index,
    #                    torch.ones(batch_size).to(x_0.device))
    F_n = feat_T.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)

    Patchs_center = x_t_xt_p_index.unsqueeze(2) - Patchs  # (B, n, K, 3)

    feat = pytorch3d.ops.knn_gather(feat_T, pi)  # (B, n, K, F)

    # use Score-Based compute grad
    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n k f -> (b n) k f"),
        flag=True,
    )  # (B*n,K,3), (B*n,K,1)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    # grad_fusion, mask = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
    #                                         True)  # (B,N,3)
    grad_weight_acc = grad_weight.sum(dim=2, keepdim=True)  # (B, n, 1, 1)
    grad_weight = grad_weight / grad_weight_acc
    grad_fusion_ = (grad_pred * grad_weight).sum(dim=2)  # (B, n, 3)
    # (B, n, 3) -> (B, N, 3)
    grad_fusion = torch.zeros_like(x_0).to(x_0.device)
    grad_fusion[:, xt_p_idx, :] = grad_fusion_

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs  # x_0 to x_t
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # x_t to x_0

    # print(dis_weight.shape)
    # print("mash mean:", mask.mean())
    # print("weight mean", grad_weight.mean())
    s2 = (self.alpha_mix + (1 - self.alpha_mix) /
          (self.train_noise * tT))**2  # (B,1,1)
    s2 = torch.clamp(s2, max=1000.0)
    # print(tT, s2)
    # loss += (mask * s2 * dis_weight *
    #          (grad_fusion - gt_offset_fusion)**2.0).sum(
    #              dim=-1).mean() * (1.0 / self.dsm_sigma)
    loss += ((mask * s2 *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma)).mean(dim=-1)

    # phase 2
    # t_s = torch.randn(batch_size).to(x_0.device) # OMG this is wrong!!
    t_s = torch.rand(batch_size).to(x_0.device)
    # tt = rearrange(t_s, "B -> B () ()")

    tt_i = (t_s * t).int()

    alpha_bar_tt = self.var_sched.alpha_bars[tt_i]
    sigma_tt = torch.sqrt((1 - alpha_bar_tt) / alpha_bar_tt).view(-1, 1, 1)
    # print("t_s", t_s)
    # print("tt_i", tt_i)
    # print("sigma_tt", sigma_tt)

    if self.train_phase2_mode_switch == 1:
      e_rand = ((1 - mask) * gt_offset_fusion + mask * grad_fusion)
    elif self.train_phase2_mode_switch == 2:
      e_rand = gt_offset_fusion
    else:
      _, _, x_0_nn = pytorch3d.ops.knn_points(x_t_a, x_0, K=1,
                                              return_nn=True)  # (B, N, 1, 3)

      x_0_nn = rearrange(x_0_nn, "B N 1 d -> B N d")
      e_rand = x_0_nn - x_t_a  # x_t to x_0

    # NOTE: evaluate GuassianNoise
    x_T_a = x_t_a.clone()
    # x_t_a = x_t_a + (1 - tt) * e_rand  # (B, N, d)
    x_t_a = x_t_a + (1 - sigma_tt / sigma_t) * e_rand  # (B, N, d)
    # print("1-sigma_tt/sigma_t", 1 - sigma_tt / sigma_t)

    feat_t = self.featureNet(x_t_a)  # (B, N, F)
    # feat_t = feat_t[:, xt_p_idx, :]  # (B, n, F)
    # x_T_xt_p_index = x_t_xt_p_index.clone()
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]

    # feat_t = self.fusion(feat_T, feat_t, x_T_a, x_t_a, t_s)
    feat_t = self.fusion(feat_T, feat_t, x_T_a, x_t_a, (tt_i / t).float())

    # F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)

    Patchs_center = x_t_xt_p_index.unsqueeze(2) - Patchs  # (B, n, K, 3)
    feat = pytorch3d.ops.knn_gather(feat_t, pi)  # (B, n, K, F)

    # use Score-Based compute grad
    # grad_pred = self.scoreNet(
    #     x=Patchs_center.view(-1, self.frame_knn, point_dim),
    #     c=rearrange(feat, "b n f -> (b n) f"),
    # ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
    #           point_dim)  # (B, n, K, 3)

    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n k f -> (b n) k f"),
        flag=True,
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)

    grad_weight_acc = grad_weight.sum(dim=2, keepdim=True)  # (B, n, 1, 1)
    grad_weight = grad_weight / grad_weight_acc
    grad_fusion_ = (grad_pred * grad_weight).sum(dim=2)  # (B, n, 3)
    # (B, n, 3) -> (B, N, 3)
    # grad_fusion = torch.zeros_like(x_0).to(x_0.device)
    grad_fusion[:, xt_p_idx, :] = grad_fusion_

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # (B, N, 3)

    # 1
    loss += ((mask *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma2)).mean(dim=-1)

    return loss

  def get_loss_v4(self, x_0, T=None):
    """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
    batch_size, point_num, point_dim = x_0.size()
    loss = 0
    t = self.var_sched.uniform_sample_t(batch_size)
    t = self.randomFunc_Tt(t, self.var_sched)

    alpha_bar_t = self.var_sched.alpha_bars[t]

    t = torch.tensor(t).to(x_0.device)
    tT = rearrange(t, "B -> B () ()") / self.var_sched.num_steps

    # NOTE: random GaussianNoise e_rand -> add noise 2 x_0
    e_rand = self.train_noise * torch.randn_like(x_0)  # (B, N, d)
    # NOTE: evaluate GuassianNoise
    x_t_a = x_0 + tT * e_rand

    feat_T = self.featureNet(x_t_a)  # (B, N, F)

    # get top point_num index
    xt_p_idx = torch.arange(self.num_train_points).to(x_0.device)
    mask = torch.zeros_like(x_0).to(x_0.device)
    mask[:, xt_p_idx, :] = 1
    # print(x_t_a.shape, xt_p_idx.shape)

    feat = feat_T[:, xt_p_idx, :]  # (B, n, F)
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]  # (B,n,3)

    # TODO: remove fusion in t=1
    # feat = self.fusion(feat, feat, x_t_xt_p_index, x_t_xt_p_index,
    #                    torch.ones(batch_size).to(x_0.device))
    F_n = feat_T.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)

    Patchs_center = Patchs - x_t_xt_p_index.unsqueeze(2)  # (B, n, K, 3)

    # feat = pytorch3d.ops.knn_gather(feat_T, pi)  # (B, n, K, F)

    # use Score-Based compute grad
    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n f -> (b n) f"),
        flag=False,
    )  # (B*n,K,3), (B*n,K,1)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)
    grad_fusion, mask_1 = grad_compute_neural(grad_pred, x_t_a, grad_weight, pi,
                                              True)  # (B,N,3)
    # grad_weight_acc = grad_weight.sum(dim=2, keepdim=True)  # (B, n, 1, 1)
    # grad_weight = grad_weight / grad_weight_acc
    # grad_fusion_ = (grad_pred * grad_weight).sum(dim=2)  # (B, n, 3)
    # # (B, n, 3) -> (B, N, 3)
    # grad_fusion = torch.zeros_like(x_0).to(x_0.device)
    # grad_fusion[:, xt_p_idx, :] = grad_fusion_

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs  # x_0 to x_t
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # x_t to x_0

    # print(dis_weight.shape)
    # print("mash mean:", mask.mean())
    # print("weight mean", grad_weight.mean())
    s2 = (self.alpha_mix + (1 - self.alpha_mix) /
          (self.train_noise * tT))**2  # (B,1,1)
    s2 = torch.clamp(s2, max=1000.0)
    # print(tT, s2)
    # loss += (mask * s2 * dis_weight *
    #          (grad_fusion - gt_offset_fusion)**2.0).sum(
    #              dim=-1).mean() * (1.0 / self.dsm_sigma)
    loss += ((mask_1 * s2 *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma)).mean(dim=-1)

    # phase 2
    t_s = torch.randn(batch_size).to(x_0.device)
    tt = rearrange(t_s, "B -> B () ()")

    if self.train_phase2_mode_switch == 1:
      e_rand = ((1 - mask) * gt_offset_fusion + mask * grad_fusion)
    elif self.train_phase2_mode_switch == 2:
      e_rand = gt_offset_fusion
    else:
      _, _, x_0_nn = pytorch3d.ops.knn_points(x_t_a, x_0, K=1,
                                              return_nn=True)  # (B, N, 1, 3)

      x_0_nn = rearrange(x_0_nn, "B N 1 d -> B N d")
      e_rand = x_0_nn - x_t_a  # x_t to x_0

    # NOTE: evaluate GuassianNoise
    x_T_a = x_t_a.clone()
    x_t_a = x_t_a + (1 - tt) * e_rand  # (B, N, d)

    feat_t = self.featureNet(x_t_a)  # (B, N, F)
    # feat_t = feat_t[:, xt_p_idx, :]  # (B, n, F)
    # x_T_xt_p_index = x_t_xt_p_index.clone()
    x_t_xt_p_index = x_t_a[:, xt_p_idx, :]

    feat_t = self.fusion(feat_T, feat_t, x_T_a, x_t_a, t_s)
    # F_n = feat.size(-1)

    # Local Patch construction
    _, pi, Patchs = pytorch3d.ops.knn_points(
        x_t_xt_p_index, x_t_a, K=self.frame_knn,
        return_nn=True)  # (B, n, K), (B, n, K, 3)

    Patchs_center = x_t_xt_p_index.unsqueeze(2) - Patchs  # (B, n, K, 3)
    feat = pytorch3d.ops.knn_gather(feat_t, pi)  # (B, n, K, F)

    # use Score-Based compute grad
    # grad_pred = self.scoreNet(
    #     x=Patchs_center.view(-1, self.frame_knn, point_dim),
    #     c=rearrange(feat, "b n f -> (b n) f"),
    # ).reshape(batch_size, len(xt_p_idx), self.frame_knn,
    #           point_dim)  # (B, n, K, 3)

    grad_pred, grad_weight = self.scoreNet(
        x=Patchs_center.view(-1, self.frame_knn, point_dim),
        c=rearrange(feat, "b n k f -> (b n) k f"),
        flag=True,
    )  # (B*n,K,3), (B*n,K,3)
    grad_pred = grad_pred.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                  point_dim)  # (B, n, K, 3)
    grad_weight = grad_weight.reshape(batch_size, len(xt_p_idx), self.frame_knn,
                                      1)  # (B, n, K, 1)

    grad_weight_acc = grad_weight.sum(dim=2, keepdim=True)  # (B, n, 1, 1)
    grad_weight = grad_weight / grad_weight_acc
    grad_fusion_ = (grad_pred * grad_weight).sum(dim=2)  # (B, n, 3)
    # (B, n, 3) -> (B, N, 3)
    # grad_fusion = torch.zeros_like(x_0).to(x_0.device)
    grad_fusion[:, xt_p_idx, :] = grad_fusion_

    _, _, clean_FusionPatchs = pytorch3d.ops.knn_points(
        x_t_a,  # (B, N, 3)
        x_0,  # (B, N, 3)
        K=self.num_clean_nbs,
        return_nn=True,
    )  # (B, N, C, 3)

    gt_offset_fusion = x_t_a.unsqueeze(2) - clean_FusionPatchs
    gt_offset_fusion = gt_offset_fusion.mean(2)
    gt_offset_fusion = -1 * gt_offset_fusion  # (B, N, 3)

    # 1
    loss += ((mask *
              (grad_fusion - gt_offset_fusion)**2.0).sum(dim=-1).sum(dim=-1) *
             (1.0 / self.dsm_sigma2)).mean(dim=-1)

    return loss

  def SB_GridEstimate(self, x_T_a, x_m_a, x_t_a, feat, B, N, K, d, F, round=10):
    index = [list(range(0 + i, N, round)) for i in range(round)]
    grad_pred = torch.zeros((B, N, K, d), device=x_T_a.device)
    grad_weight = torch.zeros((B, N, K, 1), device=x_T_a.device)
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
      if self.normalize_patch:
        Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)
      else:
        Patchs_center = Patchs
      # Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)

      gp, gw = self.scoreNet(
          x=Patchs_center.view(-1, K, d),  # (B*Ni, K, 3)
          c=rearrange(feat[:, x_r_i, :], "b n f -> (b n) f"),  # (B*Ni, F)
      )  # (B*Ni,K,3), (B*Ni,K,1)
      grad_pred[:, x_r_i, :, :] = gp.reshape(B, Ni, K, d)  # (B, Ni, K, 3)
      grad_weight[:, x_r_i, :, :] = gw.reshape(B, Ni, K, 1)  # (B, Ni, K, 1)
    return grad_pred, Patchs_idx, grad_weight

  def SB_GridEstimate_2(self,
                        x_T_a,
                        x_m_a,
                        x_t_a,
                        feat,
                        B,
                        N,
                        K,
                        d,
                        F,
                        round=10):
    index = [list(range(0 + i, N, round)) for i in range(round)]
    grad_pred = torch.zeros((B, N, d), device=x_T_a.device)
    # grad_weight = torch.zeros((B, N, K, 1), device=x_T_a.device)
    # Patchs_idx = torch.zeros((B, N, K), dtype=torch.int64, device=x_T_a.device)

    for x_r_i in index:
      x_m_a_r_i = x_m_a[:, x_r_i, :]
      Ni = len(x_r_i)
      _, pi, Patchs = pytorch3d.ops.knn_points(x_m_a_r_i,
                                               x_t_a,
                                               K=K,
                                               return_nn=True)  # (B, N, K, 3)
      # print(type(pi), pi[0, 0, 0],type(pi[0, 0, 0]))
      # Patchs_idx[:, x_r_i, :] = pi
      # Patchs_center = Patchs - x_T_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)
      # print("self.normalize_patch:", self.normalize_patch)
      if self.normalize_patch:
        Patchs_center = x_m_a_r_i.unsqueeze(2) - Patchs  # (B, Ni, K, 3)
      else:
        Patchs_center = Patchs
      # Patchs_center = Patchs - x_m_a_r_i.unsqueeze(2)  # (B, Ni, K, 3)
      feat_c = pytorch3d.ops.knn_gather(feat, pi)  # (B, Ni, K, F)
      print("feat_c:", feat_c.shape)
      gp, gw = self.scoreNet(
          x=Patchs_center.view(-1, K, d),  # (B*Ni, K, 3)
          c=rearrange(feat_c, "b n k f -> (b n) k f"),  # (B*Ni, K, F)
          flag=True,
      )  # (B*Ni,K,3), (B*Ni,K,1)
      gp = gp.reshape(B, Ni, K, d)  # (B, Ni, K, 3)
      print("gw", gw.shape)
      gw = gw.reshape(B, Ni, K, 1)  # (B, Ni, K, 1)
      gw_acc = gw.sum(dim=2, keepdim=True)  # (B, Ni, 1, 1)
      gw = gw / gw_acc
      # gw = torch.softmax(gw, dim=2)
      grad_pred[:, x_r_i, :] = (gw * gp).sum(dim=2)  # (B, Ni, 3)
      # grad_pred[:, x_r_i, :] = gp.reshape(B, Ni, K, d)  # (B, Ni, K, 3)
      # grad_weight[:, x_r_i, :, :] = gw.reshape(B, Ni, K, 1)  # (B, Ni, K, 1)
    return grad_pred

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
      score, Patchs_idx, weight = self.SB_GridEstimate(
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
                           torch.tensor([1.0] * batch_size).to(x_T.device))
      F_n = feat_T.size(-1)
      grad_pred, Patchs_idx, weight = self.SB_GridEstimate(
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

      # grad = grad_compute_func(grad_pred, x_T, x_T, Patchs_idx, self.kappa)
      grad = grad_compute_neural(grad_pred, x_T, weight, Patchs_idx, False)
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
              torch.tensor([step / num_steps] * batch_size).to(x_input.device))
          grad_pred, Patchs_idx, weight = self.SB_GridEstimate(
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
          grad_pred, Patchs_idx, weight = self.SB_GridEstimate(
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

        # grad = grad_compute_func(grad_pred, x_next, x_input, Patchs_idx,
        #                          self.kappa)
        grad = grad_compute_neural(grad_pred, x_next, weight, Patchs_idx, False)

        x_next += grad

        # traj.append(x_next.detach().clone().cpu())

    return x_next

  def get_t_list(self, sigma, num_steps=5):
    t_list = []
    T = int(self.var_sched.search_t(sigma))
    delta = T / num_steps
    t_list.append(0)
    for i in range(1, num_steps):
      t = i * delta
      t_list.append(int(t))
    t_list.append(T)
    print("sigma:", sigma)
    print("T:", T)
    print("t_list:", t_list)
    return t_list

  # org
  def sample(
      self,
      x_T,
      grad_compute_func=None,
      flexibility=0.0,
      ret_traj=False,
      sample_knn=64,
      var_sched=None,
      round=10,
      feature_fusion=True,
      is_sigma_eta_zero=True,
      eta=0,
      use_nearest_feat=False,
      break_at_step=1000,
  ):
    if var_sched == None:
      print("var_sched is None")
      var_sched = self.var_sched

    if self.tag == "ab_score":
      print("ab_score")
      if self.ctx_num_steps == 5:
        step_size = 0.46
        sb_ff = True
      else:
        step_size = 0.4
        sb_ff = True
      return self.sample_langevin_dynamics(
          x_T,
          grad_compute_func,
          flexibility,
          ret_traj,
          sample_knn=sample_knn,
          step_size=step_size,  # 0.45,0.6
          num_steps=self.ctx_num_steps,
          feature_fusion=sb_ff,
      )
    elif self.tag == "ab_gauss":
      print("ab_gauss")
      return self.sample_gauss(
          x_T,
          grad_compute_func,
          flexibility,
          ret_traj,
          sample_knn,
          var_sched,
          round,
          feature_fusion,
      )
    print("sigma:", self.ctx_noiseL)
    print("num_steps:", self.ctx_num_steps)
    t_list = self.get_t_list(sigma=self.ctx_noiseL,
                             num_steps=self.ctx_num_steps)
    T_input = t_list[-1]
    # releaseGpuCache()
    # print(x_T.shape)
    device = x_T.device
    # x_T_a = x_T_a.unsqueeze(0)  # (1, N, 3)
    batch_size, num_points, point_dim = x_T.size()

    # alpha_bar = var_sched.alpha_bars[var_sched.num_steps]
    # c0_aT = torch.sqrt(alpha_bar)
    # x_T_a = x_T_a / c0_aT
    # noiseL = var_sched.vs_sigma_acc()
    traj = {self.ctx_num_steps: x_T.clone()}
    mu = {}

    # mt = int(var_sched.num_steps / 2)

    with torch.no_grad():
      self.scoreNet.eval()
      self.featureNet.eval()
      # Feature extraction
      feat_T = self.featureNet(x_T)  # (B, N, F)
      feat_t = feat_T.clone()
      if self.tag == "Fmean":
        print("use F_mean")
        feat_acc = torch.zeros_like(feat_T)
        x_acc_a = torch.zeros_like(x_T)
        w_cnt = 0
        # sample_knn = 26
      # feat_acc = torch.zeros_like(feat_T)
      # x_acc_a = torch.zeros_like(x_T)
      # w_cnt = 0
      F_n = feat_T.size(-1)
      # a_step = 0.03
      # a_update = 1.0
      # for t in range(var_sched.num_steps, 0, -1):
      # for t in tqdm(range(var_sched.num_steps, 0, -1), desc="Sampling"):
      i_p = [
          0.1835404634475708, 0.22547364234924316, 0.2928932309150696,
          0.42291176319122314, 1.0
      ]
      for t in range(self.ctx_num_steps, 0, -1):
        # TODO: rand it with the
        # z = (
        #     torch.randn(batch_size, num_points, 1).to(device)
        #     if t > 1
        #     else torch.zeros(batch_size, num_points, 1).to(device)
        # )
        t_next = t_list[t - 1]
        t_this = t_list[t]
        # alpha = var_sched.alphas[t]
        alpha_bar_t = var_sched.alpha_bars[t_next]
        # alpha_t = var_sched.alphas[t-1]

        alpha_bar_T = var_sched.alpha_bars[t_this]
        # beta = var_sched.betas[t]
        # sigma = var_sched.get_sigmas(t, flexibility)
        # sa_b = torch.sqrt(alpha_bar)
        # c0_a_i = 1.0/c0_a
        # sa = torch.sqrt(alpha)
        # b = beta
        # sa_b_i = torch.sqrt(1 - alpha_bar)

        x_t = traj[t]
        # x_t = x_t_a * sa_b

        # TODO: change it.

        if feature_fusion:
          print("use fusion")
          # w = 1
          # feat_acc += feat_t_ * w
          # w_cnt += w
          # x_acc_a += x_t * w
          # feat_t = feat_acc / w_cnt
          # x_m = x_acc_a / w_cnt  # (B, N, 3)
          # feat_t_ = self.featureNet(x_t)  # (B, N, F)
          feat_t = self.fusion(
              feat_T, self.featureNet(x_t), x_T, x_t,
              torch.tensor([t_this / T_input] * batch_size).to(device))
          x_m = x_t
        else:
          if self.tag == "FT":
            print("use F_T")
            feat_t = feat_T
            x_m = x_T
          elif self.tag == "Ft":
            print("use F_t")
            # feat_t_ = self.featureNet(x_t)  # (B, N, F)
            feat_t = self.featureNet(x_t)
            x_m = x_t
          elif self.tag == "Fmean":
            print("use F_mean")
            feat_t = self.featureNet(x_t)
            feat_t = (feat_t + feat_T) / 2
            x_m = (x_t + x_T) / 2
            # feat_t = self.featureNet(x_t)  # (B, N, F)
          else:
            print("Error: no tag with {}".format(self.tag))
            return
        # print("mean")
        # print("feat_t:", feat_t.shape)

        # NOTE:　max-pool
        # feat_t = torch.max(feat_t, feat_t_)
        # # w = sa_b_i / sa_b / noiseL
        # # w_cnt += w

        # # x_m = (1 - w_cnt) * x_T + w_cnt * x_t
        # x_m = torch.max(x_t, x_acc_a)
        # x_acc_a = x_m.clone()
        if self.denoise_mode == 3 or self.denoise_mode == 4:
          print("use denoise_mode 3 or 4")
          grad = self.SB_GridEstimate_2(
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
        else:
          print("use denoise_mode 0 1 2")
          # -------------------------------------------------------------------------
          grad_pred, Patchs_idx, weight = self.SB_GridEstimate(
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

          if self.tag == "ab_woGF":
            del weight
            weight = torch.zeros(size=(batch_size, num_points, sample_knn, 1),
                                 device=x_T.device)
            weight[:, :, 0, :] = 1.0
            print("use ab_woGF")
          if self.tag == "ab_wGFconst":
            del weight
            weight = torch.ones(size=(batch_size, num_points, sample_knn, 1),
                                device=x_T.device)
            print("use ab_wGFconst")

          grad = grad_compute_neural(grad_pred, x_t, weight, Patchs_idx, False)
          # -------------------------------------------------------------------------

        if eta == 0.0:
          # mu_w = 1 - torch.nan_to_num(
          #     torch.sqrt((alpha - alpha_bar) / (1 - alpha_bar)),
          #     nan=0,
          #     posinf=0,
          # )
          mu_w = 1 - torch.nan_to_num(
              torch.sqrt((1 - alpha_bar_t) * alpha_bar_T /
                         ((1 - alpha_bar_T) * alpha_bar_t)),
              nan=0,
              posinf=0,
          )
          if self.tag == "ab_searchbeta" and self.ctx_num_steps == 5:
            mu_w = torch.tensor(i_p[self.ctx_num_steps - t])
          print("({} {} -> {}) eta=0 w={} cmp={}".format(
              t, t_this, t_next, mu_w.float(), 1 - torch.sqrt(
                  var_sched.get_sigma_bar(t_next) /
                  var_sched.get_sigma_bar(t_this)).float()))
          x_next = x_t + mu_w * grad

        else:
          # mu_w = torch.nan_to_num(beta / (1 - alpha_bar), nan=0, posinf=0)
          # mu_t = x_t + mu_w * grad
          # # x_next = mu_t
          # z = (torch.randn(batch_size, num_points) if t > 1 else torch.zeros(
          #     batch_size, num_points)).to(device)
          # e_norm = grad / grad.norm(p=2, dim=-1, keepdim=True)
          # e_norm = torch.where(torch.isnan(e_norm), torch.full_like(e_norm, 0),
          #                      e_norm)
          # x_next = mu_t + sigma * z.unsqueeze(dim=-1) * e_norm
          sigma = torch.sqrt((1 - alpha_bar_t) * (alpha_bar_t - alpha_bar_T) /
                             (alpha_bar_t**2 * (1 - alpha_bar_T)))
          mu_w = torch.nan_to_num(
              1 - torch.sqrt(
                  (1 - alpha_bar_t * (1 + eta * sigma**2)) * alpha_bar_T /
                  ((1 - alpha_bar_T) * alpha_bar_t)),
              nan=0,
              posinf=0,
          )
          print("({} {} -> {}) eta={} w={}".format(t, t_this, t_next, eta,
                                                   mu_w.float()))
          mu_t = x_t + mu_w * grad

          z = (torch.randn(batch_size, num_points, 3) if t > 1 else torch.zeros(
              batch_size, num_points, 3)).to(device)

          x_next = mu_t + np.sqrt(eta) * sigma * z
          print("skip")
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
        # else:
        #   mu[t] = mu_t.clone().cpu()
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
