import torch
import torch.nn as nn
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from math import ceil, pi, sqrt
from ..utils import *

from .utils.embed import PointEmbed, TimeStepEmbedding
from .utils.attention import AttenConv, AttenQK


class FusionAttenBlock(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedding(ch=self.c_dim)

    self.fusion = AttenConv(query_dim=self.c_dim,
                            context_dim=self.c_dim,
                            heads=4,
                            dim_head=16)

  def forward(self, fT, ft, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    pos_embed = self.pos_embedding(pos)  # (B,n,d) -> (B,n,c)
    t_embed = self.timestamp_embedding(t)  # (B) -> (B,c)
    t_embed = repeat(t_embed, 'b c -> b n c', n=pos_embed.shape[1])
    embed = pos_embed + t_embed  # (B,n,c)
    embed = rearrange(embed, "b n d -> (b n) () d")  # (B,n,c) -> (B*n,1,c)
    f = torch.cat([
        rearrange(fT, "b n d -> (b n) () d"),
        rearrange(ft, "b n d -> (b n) () d")
    ],
                  dim=1)  # (B*n,2,c)
    f = self.fusion(embed, context=f, mask=None)  # (B*n,1,c)
    return rearrange(f, "(b n) 1 d -> b n d", b=fT.shape[0], n=fT.shape[1])


class FusionLerpBlock(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedding(ch=self.c_dim)

    self.fusion = AttenConv(query_dim=self.c_dim,
                            context_dim=self.c_dim,
                            heads=4,
                            dim_head=12)
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    pos_embed = self.pos_embedding(pos)  # (B,n,d) -> (B,n,c)
    t_embed = self.timestamp_embedding(t)  # (B) -> (B,c)
    t_embed = repeat(t_embed, 'b c -> b n c', n=pos_embed.shape[1])
    embed = pos_embed + t_embed  # (B,n,c)
    embed = rearrange(embed, "b n c -> (b n) () c")  # (B,n,c) -> (B*n,1,c)
    # f = torch.cat([
    #     rearrange(fT, "b n d -> (b n) () d"),
    #     rearrange(ft, "b n d -> (b n) () d")
    # ],
    #               dim=1)  # (B*n,2,c)
    # f = self.fusion(embed, context=f, mask=None)  # (B*n,1,c)
    # return rearrange(f, "(b n) 1 d -> b n d", b=fT.shape[0], n=fT.shape[1])
    f = torch.cat([
        rearrange(fT, "b n c -> (b n) () c"),
        rearrange(ft, "b n c -> (b n) () c")
    ],
                  dim=1)  # (B*n,2,c)
    e = self.fusion(embed, context=f, mask=None)  # (B*n,1,c)
    e = rearrange(e, "(b n) 1 c -> b n c", b=fT.shape[0], n=fT.shape[1])
    return fT * e + ft * (1 - e)


    # embed = rearrange(embed, "b n c -> (b n) () c")  # (B,n,c) -> (B*n,1,c)
    # f = torch.cat([
    #     rearrange(fT, "b n c -> b c n"),
    #     rearrange(ft, "b n c -> b c n"),
    #     rearrange(embed, "b n c -> b c n")
    # ],
    #               dim=1)  # (B,3*c,n)
    # f = self.conv_mlp(f)  # (B,3*c,n) -> (B,c,n)
    # print("o", f.shape)
    # return rearrange(f, "b c n -> b n c")
    # embed = self.conv_mlp(rearrange(embed, "b n c -> b c n"))
    # embed = rearrange(embed, "b c n -> b n c")
    # return fT * embed + ft * (1 - embed)
class FusionBlock(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    self.c_dim = c_dim

  def forward(self, fT, ft, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """

    return (fT + ft) / 2


class FusionQKBlock(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedding(ch=self.c_dim)

    self.fusion = AttenQK(query_dim=self.c_dim)
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    pos_embed = self.pos_embedding(pos)  # (B,n,d) -> (B,n,c)
    t_embed = self.timestamp_embedding(t)  # (B) -> (B,c)
    t_embed = repeat(t_embed, 'b c -> b n c', n=pos_embed.shape[1])
    embed = pos_embed + t_embed  # (B,n,c)
    embed = rearrange(embed, "b n c -> (b n) () c")  # (B,n,c) -> (B*n,1,c)
    f = torch.cat([
        rearrange(fT, "b n c -> (b n) () c"),
        rearrange(ft, "b n c -> (b n) () c")
    ],
                  dim=1)  # (B*n,2,c)
    e = self.fusion(embed, context=f)  # (B*n,1,c)
    e = rearrange(e, "(b n) 1 c -> b n c", b=fT.shape[0], n=fT.shape[1])
    return fT * e + ft * (1 - e)


class FusionQKBlock_2(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
                                  nn.ReLU())

    from .utils.attention import AttenQKlinear
    self.fusion = AttenQKlinear(query_dim=self.c_dim)
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = pos.shape
    pos_embed = self.pos_embedding(pos)  # (B,n,d) -> (B,n,c)

    # print(t.shape)
    t_embed = self.timestamp_embedding(t)  # (B) -> (B,c)
    # print(t_embed.shape)
    t_embed = repeat(t_embed, 'b c -> b n c', n=pos_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
    # print(pos_embed.shape, t_embed.shape)
    embed = torch.cat([pos_embed, t_embed], dim=2)  # (B,n,c)
    embed = self.reduce_e(embed.permute(0, 2, 1)).permute(0, 2, 1)
    embed = rearrange(embed, "b n c -> (b n) () c")  # (B,n,c) -> (B*n,1,c)
    f = torch.cat([
        rearrange(fT, "b n c -> (b n) () c"),
        rearrange(ft, "b n c -> (b n) () c")
    ],
                  dim=1)  # (B*n,2,c)
    e = self.fusion(embed, context=f)  # (B*n,1,c)
    e = rearrange(e, "(b n) 1 c -> b n c", b=fT.shape[0], n=fT.shape[1])
    return fT * e + ft * (1 - e)


class FusionLerpBlock_2(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
                                  nn.ReLU())

    from .utils.attention import AttenLinear
    self.fusion = AttenLinear(query_dim=self.c_dim)
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, pos, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      pos: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = pos.shape
    pos_embed = self.pos_embedding(pos)  # (B,n,d) -> (B,n,c)
    t_embed = self.timestamp_embedding(t)  # (B) -> (B,c)
    t_embed = repeat(t_embed, 'b c -> b n c', n=pos_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
    embed = torch.cat([pos_embed, t_embed], dim=2)  # (B,n,c)
    embed = self.reduce_e(embed.permute(0, 2, 1)).permute(0, 2, 1)
    embed = rearrange(embed, "b n c -> (b n) () c")  # (B,n,c) -> (B*n,1,c)
    f = torch.cat([
        rearrange(fT, "b n c -> (b n) () c"),
        rearrange(ft, "b n c -> (b n) () c")
    ],
                  dim=1)  # (B*n,2,c)
    e = self.fusion(embed, context=f)  # (B*n,1,c)
    e = rearrange(e, "(b n) 1 c -> b n c", b=fT.shape[0], n=fT.shape[1])
    print(torch.mean(e), torch.var(e))
    return fT * e + ft * (1 - e)


class FusionLerpBlock_0615(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
                                  nn.ReLU())

    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim), nn.ReLU())
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.Sigmoid())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    xT_embed = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(embed.permute(0, 2, 1)).permute(
        0, 2, 1)  # (B,2*n,2c) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, xT_embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, xt_embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return fT * e + ft * (1 - e)


class FusionLerpBlock_0618(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_embedding = PointEmbed(dim=self.c_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
                                  nn.ReLU())

    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim), nn.ReLU())
    from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                SoftClamp())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    xT_embed = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(embed.permute(0, 2, 1)).permute(
        0, 2, 1)  # (B,2*n,2c) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, xT_embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, xt_embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return fT * e + ft * (1 - e)


class FusionLerpBlock_0703(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1), nn.ReLU())

    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim), nn.ReLU())
    from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                SoftClamp())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    xT_embed = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(embed.permute(0, 2, 1)).permute(
        0, 2, 1)  # (B,2*n,2c) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, xT_embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, xt_embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return fT * e + ft * (1 - e)


class FusionLerpBlock_0703v2(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1),
        nn.ReLU(inplace=True))

    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                             nn.ReLU(inplace=True))
    from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.Sigmoid())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_e = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    # xT_e = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    # embed = torch.cat([xT_e, xt_e], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(xt_e.permute(0, 2,
                                       1)).permute(0, 2,
                                                   1)  # (B,n,2c) -> (B,n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return fT * e + ft * (1 - e)


class FusionLerpBlock_0703_mo(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1),
        nn.ReLU(inplace=True))

    # self.trans = FCLayer(self.c_dim, self.c_dim, bias=True, activation=None)
    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, 256),
                             nn.ReLU(inplace=True), nn.Linear(256, self.c_dim),
                             nn.ReLU(inplace=True))
    # from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.Tanh())
    # self.out = nn.Sequential(nn.Linear(self.c_dim, self.c_dim), nn.Dropout(0.0))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_e = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    # xT_e = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    # embed = torch.cat([xT_e, xt_e], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(xt_e.permute(0, 2,
                                       1)).permute(0, 2,
                                                   1)  # (B,n,2c) -> (B,n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    # fT_trans, ft_trans = torch.split(self.trans(torch.cat([fT, ft], dim=1)),
    #                                  [n, n],
    #                                  dim=1)
    FxT = torch.cat([fT, embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    return fT * e + ft * (1 - e)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    # return torch.cat([(fT * e).unsqueeze(2), (ft * (1 - e)).unsqueeze(2)],
    #                  dim=2).sum(dim=2)
    # Fx_e = torch.cat([fT_trans * Fx_eT, ft_trans * Fx_et], dim=2)  #(B,n,2c)
    # return self.out(Fx_e.reshape(-1, 2 * self.c_dim)).reshape(B, n, self.c_dim)


class FusionLerpBlock_0703_mo2l(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Sequential(
        nn.Linear(3, 32, bias=False), nn.ReLU(inplace=True),
        nn.Linear(32, self.pos_e_dim, bias=False))
    # self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1),
        nn.ReLU(inplace=True))

    # self.trans = FCLayer(self.c_dim, self.c_dim, bias=True, activation=None)
    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, 256),
                             nn.ReLU(inplace=True), nn.Linear(256, self.c_dim),
                             nn.ReLU(inplace=True))
    # from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.c_dim, self.c_dim), nn.Tanh())
    # self.out = nn.Sequential(nn.Linear(self.c_dim, self.c_dim), nn.Dropout(0.0))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_e = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    # xT_e = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    # embed = torch.cat([xT_e, xt_e], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(xt_e.permute(0, 2,
                                       1)).permute(0, 2,
                                                   1)  # (B,n,2c) -> (B,n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    # fT_trans, ft_trans = torch.split(self.trans(torch.cat([fT, ft], dim=1)),
    #                                  [n, n],
    #                                  dim=1)
    FxT = torch.cat([fT, embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    return fT * e + ft * (1 - e)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    # return torch.cat([(fT * e).unsqueeze(2), (ft * (1 - e)).unsqueeze(2)],
    #                  dim=2).sum(dim=2)
    # Fx_e = torch.cat([fT_trans * Fx_eT, ft_trans * Fx_et], dim=2)  #(B,n,2c)
    # return self.out(Fx_e.reshape(-1, 2 * self.c_dim)).reshape(B, n, self.c_dim)


class FusionLerpBlock_0703_mo2lv2(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    # self.pos_embedding = nn.Sequential(
    #     nn.Linear(3, 32, bias=False), nn.ReLU(inplace=True),
    #     nn.Linear(32, self.pos_e_dim, bias=False))
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1),
        nn.LeakyReLU(inplace=True))

    # self.trans = FCLayer(self.c_dim, self.c_dim, bias=True, activation=None)
    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, 256),
                             nn.LeakyReLU(inplace=True), nn.Linear(256, 256),
                             nn.LeakyReLU(inplace=True),
                             nn.Linear(256, self.c_dim),
                             nn.LeakyReLU(inplace=True))
    # from .utils.functionBlcok import SoftClamp
    # self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
    #                             nn.LeakyReLU(inplace=True),
    #                             nn.Linear(self.c_dim, self.c_dim), nn.Tanh())
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.Tanh())
    # self.out = nn.Sequential(nn.Linear(self.c_dim, self.c_dim), nn.Dropout(0.0))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_e = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    # xT_e = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    # embed = torch.cat([xT_e, xt_e], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(xt_e.permute(0, 2,
                                       1)).permute(0, 2,
                                                   1)  # (B,n,2c) -> (B,n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    # fT_trans, ft_trans = torch.split(self.trans(torch.cat([fT, ft], dim=1)),
    #                                  [n, n],
    #                                  dim=1)
    FxT = torch.cat([fT, embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    e = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    return fT * e + ft * (1 - e)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    # return torch.cat([(fT * e).unsqueeze(2), (ft * (1 - e)).unsqueeze(2)],
    #                  dim=2).sum(dim=2)
    # Fx_e = torch.cat([fT_trans * Fx_eT, ft_trans * Fx_et], dim=2)  #(B,n,2c)
    # return self.out(Fx_e.reshape(-1, 2 * self.c_dim)).reshape(B, n, self.c_dim)


class FusionLerpBlock_0705(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Linear(self.c_dim + self.pos_e_dim, self.c_dim), nn.ReLU())

    # from .utils.attention import AttenLinear
    # self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim * 2),
    #                          nn.ReLU())
    # from .utils.functionBlcok import SoftClamp
    # self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
    #                             nn.Sigmoid())
    self.fusion = nn.Sequential(nn.BatchNorm1d(self.c_dim * 4), nn.ReLU(),
                                nn.Conv1d(self.c_dim * 4, self.c_dim * 2, 1),
                                nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
                                nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    xt_e = self.pos_embedding(xt)  # (B,n,d) -> (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_e, t_embed], dim=2)  # (B,n,2c)
    xT_embed = torch.cat([xt_e, T_embed], dim=2)  # (B,n,2c)
    # print(xt_embed.shape, xT_embed.shape)
    embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(embed)  # (B,2*n,2c) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, xT_embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, xt_embed], dim=2)  #(B,n,2c)
    # FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    # Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,2c)
    # Fx_eT, Fx_et = torch.split(FxTt, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([FxT, Fxt], dim=2)  #(B,n,4c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return fT + self.fusion(Fx_e.transpose(1, 2)).transpose(
        1, 2)  #(B,n,4c)->(B,n,c)


class FusionLerpBlock_0705v2(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Linear(self.c_dim + self.pos_e_dim, self.c_dim), nn.ReLU())

    # from .utils.attention import AttenLinear
    # self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim * 2),
    #                          nn.ReLU())
    # from .utils.functionBlcok import SoftClamp
    # self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
    #                             nn.Sigmoid())
    from .utils.attention import AttenLinear_
    self.attn = AttenLinear_(query_dim=self.c_dim)
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    xt_e = self.pos_embedding(xt)  # (B,n,d) -> (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_e, t_embed], dim=2)  # (B,n,2c)
    xT_embed = torch.cat([xt_e, T_embed], dim=2)  # (B,n,2c)
    # print(xt_embed.shape, xT_embed.shape)
    embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(embed)  # (B,2*n,2c) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)

    fT, ft, xT_embed, xt_embed = map(
        lambda t: rearrange(t, "b n c -> (b n) () c"),
        [fT, ft, xT_embed, xt_embed])
    q = torch.cat([xT_embed, xt_embed], dim=1)
    kv = torch.cat([fT, ft], dim=1)
    _, ft_e = self.attn(q, kv).chunk(2, dim=1)  #(B*n,1,c)
    return rearrange(ft_e, "(b n) 1 c -> b n c", b=B, n=n)


class FusionLerpBlock_0705v3(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Linear(self.c_dim + self.pos_e_dim, self.c_dim), nn.ReLU())

    # from .utils.attention import AttenLinear
    # self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim * 2),
    #                          nn.ReLU())
    # from .utils.functionBlcok import SoftClamp
    # self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
    #                             nn.Sigmoid())
    # self.reduce_w = nn.Sequential(nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
    #                               nn.Sigmoid())
    self.reduce_w = nn.Sequential(nn.Conv1d(self.c_dim * 2, self.c_dim, 1),
                                  nn.Sigmoid())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())
    # self.out_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    xt_e = self.pos_embedding(xt)  # (B,n,d) -> (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_embed = torch.cat([xt_e, t_embed], dim=2)  # (B,n,2c)
    # xT_embed = torch.cat([xt_e, T_embed], dim=2)  # (B,n,2c)
    # print(xt_embed.shape, xT_embed.shape)
    # embed = torch.cat([xT_embed, xt_embed], dim=1)  # (B,2*n,2c)
    t_embed = self.reduce_e(xt_embed)  # (B,2*n,2c) -> (B,2*n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)

    eT = torch.cat([fT, t_embed], dim=2)  #(B,n,2c)
    et = torch.cat([ft, t_embed], dim=2)  #(B,n,2c)
    eT = self.reduce_w(eT.transpose(1, 2)).transpose(1, 2)
    et = self.reduce_w(et.transpose(1, 2)).transpose(1, 2)
    eSum = eT + et + 1e-8
    eT, et = eT / eSum, et / eSum
    return fT * eT + ft * et


class FusionLerpBlock_0731v1(nn.Module):

  def __init__(self, c_dim):
    super().__init__()
    from .utils.embed import TimeStepEmbedder_2
    self.c_dim = c_dim
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(3, self.pos_e_dim)
    self.timestamp_embedding = TimeStepEmbedder_2(ch=self.c_dim, dim=256)
    self.reduce_e = nn.Sequential(
        nn.Conv1d(self.c_dim + self.pos_e_dim, self.c_dim, 1), nn.ReLU())

    # from .utils.attention import AttenLinear
    self.mlp = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim), nn.ReLU())
    # from .utils.functionBlcok import SoftClamp
    self.fusion = nn.Sequential(nn.Linear(self.c_dim * 2, self.c_dim),
                                nn.ReLU())
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim * 2, 1),
    #                               nn.BatchNorm1d(self.c_dim * 2), nn.ReLU(),
    #                               nn.Conv1d(self.c_dim * 2, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim * 3, self.c_dim, 1))
    # self.conv_mlp = nn.Sequential(nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.BatchNorm1d(self.c_dim), nn.Sigmoid(),
    #                               nn.Conv1d(self.c_dim, self.c_dim, 1),
    #                               nn.Sigmoid())

  def forward(self, fT, ft, xT, xt, t):
    """
    input:
      fT: (b, npoints, dim)
      ft: (b, npoints, dim)
      xT: (b, npoints, 3)
      xt: (b, npoints, 3)
      t: (b)
    output:
      (b, npoints, dim)
    """
    # print(fT.shape, ft.shape, pos.shape, t.shape)
    # print("i", fT.shape)
    B, n, _ = xT.shape
    pos = torch.cat([xT, xt], dim=1)  # (B,2*n,3)
    pos_embed = self.pos_embedding(pos)  # (B,2*n,d) -> (B,2*n,c)
    xT_embed, xt_embed = torch.split(pos_embed, [n, n], dim=1)  # (B,n,c)
    T = torch.ones_like(t)
    Tt = torch.cat([T, t], dim=0)  # (2B)
    time_embed = self.timestamp_embedding(Tt)  # (2B) -> (2B,c)
    time_embed = repeat(time_embed, 'b c -> b n c', n=n)
    T_embed, t_embed = torch.split(time_embed, [B, B], dim=0)  # (B,c)
    # t_embed = repeat(t_embed, 'b c -> b n c', n=xT_embed.shape[1])
    if t_embed.shape[0] == 1 and B > 1:
      t_embed = repeat(t_embed, '() n c -> b n c', b=B)
      T_embed = repeat(T_embed, '() n c -> b n c', b=B)

    xt_e = torch.cat([xt_embed, t_embed], dim=2)  # (B,n,2c)
    # xT_e = torch.cat([xT_embed, T_embed], dim=2)  # (B,n,2c)
    # embed = torch.cat([xT_e, xt_e], dim=1)  # (B,2*n,2c)
    embed = self.reduce_e(xt_e.permute(0, 2,
                                       1)).permute(0, 2,
                                                   1)  # (B,n,2c) -> (B,n,c)
    # xT_embed, xt_embed = torch.split(embed, [n, n], dim=1)  # (B,n,c)
    FxT = torch.cat([fT, embed], dim=2)  #(B,n,2c)
    Fxt = torch.cat([ft, embed], dim=2)  #(B,n,2c)
    FxTt = torch.cat([FxT, Fxt], dim=1)  #(B,2*n,2c)
    Fx_e = self.mlp(FxTt)  #(B,2*n,2c)->(B,2*n,c)
    Fx_eT, Fx_et = torch.split(Fx_e, [n, n], dim=1)
    # v2
    # return fT *Fx_eT + ft * Fx_et
    Fx_e = torch.cat([Fx_eT, Fx_et], dim=2)  #(B,n,2c)
    feat_fused = self.fusion(Fx_e)  #(B,n,2c)->(B,n,c)
    # Fx_eAns = Fx_eT + Fx_et
    # v3
    return feat_fused
