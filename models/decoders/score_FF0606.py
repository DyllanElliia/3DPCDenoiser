import torch
import torch.nn as nn
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import pytorch3d.ops

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from math import ceil, pi, sqrt

from .utils.attention import *
from .utils.embed import PointEmbed


def exists(v):
  return v is not None


def default(v, d):
  return v if exists(v) else d


def first(it):
  return it[0]


def divisible_by(num, den):
  return (num % den) == 0


def is_odd(n):
  return not divisible_by(n, 2)


def is_empty(l):
  return len(l) == 0


def is_tensor_empty(t: Tensor):
  return t.numel() == 0


def set_module_requires_grad_(module: Module, requires_grad: bool):
  for param in module.parameters():
    param.requires_grad = requires_grad


def l1norm(t):
  return F.normalize(t, dim=-1, p=1)


def l2norm(t):
  return F.normalize(t, dim=-1, p=2)


def safe_cat(tensors, dim):
  tensors = [*filter(exists, tensors)]

  if len(tensors) == 0:
    return None
  elif len(tensors) == 1:
    return first(tensors)

  return torch.cat(tensors, dim=dim)


def pad_at_dim(t, padding, dim=-1, value=0):
  ndim = t.ndim
  right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
  zeros = (0, 0) * right_dims
  return F.pad(t, (*zeros, *padding), value=value)


def pad_to_length(t, length, dim=-1, value=0, right=True):
  curr_length = t.shape[dim]
  remainder = length - curr_length

  if remainder <= 0:
    return t

  padding = (0, remainder) if right else (remainder, 0)
  return pad_at_dim(t, padding, dim=dim, value=value)


# resnet block


class PixelNorm(nn.Module):

  def __init__(self, dim, eps=1e-4):
    super().__init__()
    self.dim = dim
    self.eps = eps

  def forward(self, x):
    dim = self.dim
    return F.normalize(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])


class SimpleNorm(nn.Module):

  def __init__(self, hidden_dim) -> None:
    super().__init__()
    self.bn_out = nn.BatchNorm1d(hidden_dim)
    self.actvn_out = nn.ReLU()

  def forward(self, x):
    return self.actvn_out(self.bn_out(x))


class PreNorm(nn.Module):

  def __init__(self, dim, fn, context_dim=None):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)
    self.norm_context = nn.LayerNorm(context_dim) if exists(
        context_dim) else None

  def forward(self, x, **kwargs):
    x = self.norm(x)

    if exists(self.norm_context):
      context = kwargs['context']
      normed_context = self.norm_context(context)
      kwargs.update(context=normed_context)

    return self.fn(x, **kwargs)


class SqueezeExcite(nn.Module):

  def __init__(self, dim, reduction_factor=4, min_dim=16):
    super().__init__()
    dim_inner = max(dim // reduction_factor, min_dim)

    self.net = nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.SiLU(),
        nn.Linear(dim_inner, dim),
        nn.Sigmoid(),
        Rearrange("b c -> b c 1"),
    )

  def forward(self, x, mask=None):
    if exists(mask):
      x = x.masked_fill(~mask, 0.0)

      num = reduce(x, "b c n -> b c", "sum")
      den = reduce(mask.float(), "b 1 n -> b 1", "sum")
      avg = num / den.clamp(min=1e-5)
    else:
      avg = reduce(x, "b c n -> b c", "mean")

    return x * self.net(avg)


class Block(nn.Module):

  def __init__(self, dim, dim_out=None, kernel_size=1, dropout=0.0):
    super().__init__()
    dim_out = default(dim_out, dim)

    # self.proj = nn.Conv1d(dim, dim_out, kernel_size, padding=1)
    self.proj = nn.Conv1d(dim, dim_out, kernel_size)
    # self.norm = PixelNorm(dim=1)
    self.norm = SimpleNorm(dim_out)
    # self.dropout = nn.Dropout(dropout)
    # self.act = nn.SiLU()

  def forward(self, x):
    # print("x i:", x.shape)
    x = self.proj(self.norm(x))
    # print("x o:", x.shape)
    # x = self.norm(x)
    # x = self.act(x)
    # x = self.dropout(x)

    return x


class ResnetBlock(nn.Module):

  def __init__(self, dim, dim_out=None, *, dropout=0.0):
    super().__init__()
    dim_out = default(dim_out, dim)
    self.block1 = Block(dim, dim_out, dropout=dropout)
    self.block2 = Block(dim_out, dim_out, dropout=dropout)
    self.excite = SqueezeExcite(dim_out)
    self.residual_conv = (nn.Conv1d(dim, dim_out, 1)
                          if dim != dim_out else nn.Identity())
    # self.res_norm = PixelNorm(dim=1)
    self.res_norm = SimpleNorm(dim)

  def forward(self, x):
    res = self.residual_conv(self.res_norm(x))
    # res = self.residual_conv(x)
    h = self.block1(x)
    # print(h.shape, res.shape)
    h = self.block2(h)
    # h = self.excite(h, mask=mask)
    # print(h.shape, res.shape)
    return h + res


class ResnetBlockC(nn.Module):

  def __init__(self, c_dim, dim, dim_out=None, *, dropout=0.0):
    super().__init__()
    dim_out = default(dim_out, dim)
    self.block1 = Block(dim, dim_out, dropout=dropout)
    self.block2 = Block(dim_out, dim_out, dropout=dropout)
    # self.block3 = Block(dim_out, dim_out, dropout=dropout)
    self.excite = SqueezeExcite(dim_out)
    self.residual_conv = nn.Conv1d(c_dim, dim_out, 1)

    # self.res_norm = PixelNorm(dim=1)
    # self.res_norm = SimpleNorm(c_dim)
    if dim == dim_out:
      self.shortcut = None
    else:
      self.shortcut = nn.Conv1d(dim, dim_out, 1, bias=False)

  def forward(self, x, c=None):
    if c is None:
      c = x
    # res = self.residual_conv(self.res_norm(c))
    res = self.residual_conv(c)
    h = self.block1(x)
    # print(h.shape, res.shape)
    h = self.block2(h)
    # h = self.block3(h, mask=mask)
    h = self.excite(h)
    # print(h.shape, res.shape)
    if self.shortcut is not None:
      x_s = self.shortcut(x)
    else:
      x_s = x
    return x_s + h + res


class ScoreNet(nn.Module):

  def __init__(
      self,
      z_dim,
      dim,
      out_dim,
      hidden_size,
      num_blocks,
  ):
    """
        Args:
            z_dim:   Dimension of context vectors.
            dim:     Point dimension.
            out_dim: Gradient dim.
            hidden_size:   Hidden states dim.
        """
    super().__init__()

    self.z_dim = z_dim
    self.dim = dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_blocks = num_blocks

    self.point_embeding = PointEmbed(dim=self.z_dim)
    self.neighbor_fusion = AttenLinear(self.z_dim)

    # Input = Conditional = zdim (code) + dim (xyz)
    # c_dim = 2*z_dim

    res_layer = []
    # self.reduce_mlp = nn.Sequential(nn.Conv1d(self.z_dim * 2, self.z_dim, 1),
    #                                 nn.BatchNorm1d(self.z_dim), nn.ReLU())
    # res_layer.append(self.reduce_mlp)
    res_layer += [
        nn.Conv1d(self.z_dim * 2, self.z_dim, 1),
        nn.BatchNorm1d(self.z_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks - 1):
      res_layer += [
          ResnetBlockC(self.z_dim, self.z_dim),
          nn.BatchNorm1d(self.z_dim),
          nn.ReLU()
      ]
    res_layer.append(nn.Conv1d(self.z_dim, self.dim, 1))
    self.conv_mlp = nn.Sequential(*res_layer)
    # self.to_out = nn.Linear(self.z_dim, self.dim)

  def forward(self, x, c, k=32):
    """
        :param x: (B, n, d) Input coordinate (xyz)
        :param c: (B, n, c) Shape latent code
        :return: (B, n, d) Gradient (self.dim dimension)
        """
    B, n, _ = x.shape
    _, pi, P = pytorch3d.ops.knn_points(x, x, K=k)
    C = pytorch3d.ops.knn_gather(c, pi)  # (B,n,k,c)
    C = rearrange(C, 'B n k c -> (B n) k c')

    # embedding p:(B,n,d) -> (B,n,c) -> (B*n, 1, c)
    p_emb = self.point_embeding(x)
    p_emb = rearrange(p_emb, 'B n c -> (B n) 1 c')

    # p_c: cross-atten(p_emb,C) -> (B*n,1,c) -> (B,n,c)
    p_c = self.neighbor_fusion(p_emb, C)
    p_c = rearrange(p_c, '(B n) 1 c -> B n c', n=n)

    # encode gradient from cat[p_emb,p_c]: (B,n,2c) -> do?(B,n,c) -> (B,n,d)
    p_emb = rearrange(p_emb, '(B n) 1 c -> B n c', n=n)
    p_ec = torch.cat([p_emb, p_c], dim=-1)
    p_ec = rearrange(p_ec, 'B n c -> B c n')
    # p_ec = self.reduce_mlp(p_ec)
    return rearrange(self.conv_mlp(p_ec), 'B d n -> B n d')
