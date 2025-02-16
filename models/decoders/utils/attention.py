'''
Author: DyllanElliia
Date: 2024-05-07 13:52:18
LastEditors: DyllanElliia
LastEditTime: 2024-05-07 13:55:18
Description: 
'''
from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

from .define import default, exists, cache_fn


class Attention(nn.Module):

  def __init__(self,
               query_dim,
               context_dim=None,
               heads=8,
               dim_head=64,
               dropout=0.0):
    super().__init__()
    inner_dim = dim_head * heads
    context_dim = default(context_dim, query_dim)
    self.scale = dim_head**-0.5
    self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

    self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                nn.Dropout(dropout))

  def forward(self, x, context=None, mask=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    h = self.heads

    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    context = default(context, x)
    k, v = self.to_kv(context).chunk(2, dim=-1)  # (B,m,d) -> (B,m,Di), (B,m,Di)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                  (q, k, v))
    # (B*h,n,Di/h) op (B*h,m,Di/h) -> (B*h,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
      mask = rearrange(mask, 'b ... -> b (...)')
      max_neg_value = -torch.finfo(sim.dtype).max
      mask = repeat(mask, 'b j -> (b h) () j', h=h)
      sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B*h,n,m) -> (B*h,n,m)
    # (B*h,n,m) op (B*h,m,Di/h) -> (B*h,n,Di/h)
    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class AttenConv(nn.Module):

  def __init__(self,
               query_dim,
               context_dim=None,
               heads=8,
               dim_head=64,
               dropout=0.0):
    super().__init__()
    inner_dim = dim_head * heads
    context_dim = default(context_dim, query_dim)
    self.scale = dim_head**-0.5
    self.heads = heads

    self.to_q = nn.Conv1d(query_dim, inner_dim, 1)
    self.to_kv = nn.Conv1d(context_dim, inner_dim * 2, 1)

    self.to_out = nn.Conv1d(inner_dim, query_dim, 1)

  def forward(self, x, context=None, mask=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    h = self.heads

    x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    context = rearrange(context, 'b m d -> b d m')
    k, v = rearrange(self.to_kv(context), "b d m -> b m d").chunk(
        2, dim=-1)  # (B,m,d) -> (B,m,Di), (B,m,Di)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                  (q, k, v))
    # (B*h,n,Di/h) op (B*h,m,Di/h) -> (B*h,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
      mask = rearrange(mask, 'b ... -> b (...)')
      max_neg_value = -torch.finfo(sim.dtype).max
      mask = repeat(mask, 'b j -> (b h) () j', h=h)
      sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B*h,n,m) -> (B*h,n,m)
    # (B*h,n,m) op (B*h,m,Di/h) -> (B*h,n,Di/h)
    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = rearrange(out, 'b d n -> b n d')
    return rearrange(self.to_out(out), "b d n -> b n d")


class AttenQK(nn.Module):

  def __init__(
      self,
      query_dim,
      # context_dim=None,
      #  heads=8,
  ):
    super().__init__()
    inner_dim = query_dim
    context_dim = query_dim
    self.scale = inner_dim**-0.5
    # self.heads = heads

    self.to_q = nn.Conv1d(query_dim, inner_dim, 1)
    self.to_k = nn.Conv1d(context_dim, inner_dim, 1)

    self.to_out = nn.Conv1d(inner_dim, query_dim, 1)

  def forward(self, x, context=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    # h = self.heads

    x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    v = context
    # print(context.shape)
    context = rearrange(context, 'b m d -> b d m')
    k = rearrange(self.to_k(context), "b d m -> b m d")  # (B,m,d) -> (B,m,Di)
    # print(k.shape)

    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
    #               (q, k, v))
    # (B,n,Di) op (B,m,Di) -> (B,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    # print(sim.shape)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B,n,m) -> (B,n,m)
    # print(attn.shape, v.shape)
    # (B,n,m) op (B,m,Di) -> (B,n,Di)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out
    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # out = rearrange(out, 'b d n -> b n d')
    # return rearrange(self.to_out(out), "b d n -> b n d")


class AttenQKlinear(nn.Module):

  def __init__(
      self,
      query_dim,
      # context_dim=None,
      #  heads=8,
  ):
    super().__init__()
    inner_dim = query_dim
    context_dim = query_dim
    self.scale = inner_dim**-0.5
    # self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_k = nn.Linear(context_dim, inner_dim, bias=False)

    # self.to_out = nn.Conv1d(inner_dim, query_dim, 1)

  def forward(self, x, context=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    # h = self.heads

    # x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    # q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    v = context
    # print(context.shape)
    # context = rearrange(context, 'b m d -> b d m')
    k = self.to_k(context)  # (B,m,d) -> (B,m,Di)
    # print(k.shape)

    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
    #               (q, k, v))
    # (B,n,Di) op (B,m,Di) -> (B,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    # print(sim.shape)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B,n,m) -> (B,n,m)
    # print(attn.shape, v.shape)
    # (B,n,m) op (B,m,Di) -> (B,n,Di)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # out = rearrange(out, 'b d n -> b n d')
    # return rearrange(self.to_out(out), "b d n -> b n d")
class AttenLinear(nn.Module):

  def __init__(
      self,
      query_dim,
      # context_dim=None,
      #  heads=8,
  ):
    super().__init__()
    inner_dim = query_dim
    context_dim = query_dim
    self.scale = inner_dim**-0.5
    # self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_k = nn.Linear(context_dim, inner_dim, bias=False)

    self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

  def forward(self, x, context=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    # h = self.heads

    # x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    # q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    v = context
    # print(context.shape)
    # context = rearrange(context, 'b m d -> b d m')
    k = self.to_k(context)  # (B,m,d) -> (B,m,Di)
    # print(k.shape)

    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
    #               (q, k, v))
    # (B,n,Di) op (B,m,Di) -> (B,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    # print(sim.shape)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B,n,m) -> (B,n,m)
    # print(attn.shape, v.shape)
    # (B,n,m) op (B,m,Di) -> (B,n,Di)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return self.to_out(out)
    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # out = rearrange(out, 'b d n -> b n d')
    # return rearrange(self.to_out(out), "b d n -> b n d")


class AttenLinear_(nn.Module):

  def __init__(
      self,
      query_dim,
      # context_dim=None,
      #  heads=8,
  ):
    super().__init__()
    inner_dim = query_dim
    context_dim = query_dim
    self.scale = inner_dim**-0.5
    # self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

    self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

  def forward(self, x, context=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    # h = self.heads

    # x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    # q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    # v = context
    # print(context.shape)
    # context = rearrange(context, 'b m d -> b d m')
    k, v = self.to_kv(context).chunk(2, dim=-1)  # (B,m,d) -> (B,m,Di)
    # print(k.shape)

    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
    #               (q, k, v))
    # (B,n,Di) op (B,m,Di) -> (B,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    # print(sim.shape)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B,n,m) -> (B,n,m)
    # print(attn.shape, v.shape)
    # (B,n,m) op (B,m,Di) -> (B,n,Di)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return self.to_out(out)
    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # out = rearrange(out, 'b d n -> b n d')
    # return rearrange(self.to_out(out), "b d n -> b n d")


class AttenOne(nn.Module):

  def __init__(
      self,
      query_dim,
      # context_dim=None,
      #  heads=8,
  ):
    super().__init__()
    inner_dim = query_dim
    context_dim = query_dim
    self.scale = inner_dim**-0.5
    # self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

    self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

  def forward(self, x, context=None):
    # input:
    #   x: (B,n,d)
    #   c: (B,m,d)
    # output:
    #   x: (B,n,d)
    # h = self.heads

    # x = rearrange(x, 'b n d -> b d n')
    q = self.to_q(x)  # (B,n,d) -> (B,n,Di)
    # q = rearrange(q, 'b d n -> b n d')
    context = default(context, x)
    # v = context
    # print(context.shape)
    # context = rearrange(context, 'b m d -> b d m')
    k, v = self.to_kv(context).chunk(2, dim=-1)  # (B,m,d) -> (B,m,Di)
    # print(k.shape)

    # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
    #               (q, k, v))
    # (B,n,Di) op (B,m,Di) -> (B,n,m) * scale
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    # print(sim.shape)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)  # (B,n,m) -> (B,n,m)
    # print(attn.shape, v.shape)
    # (B,n,m) op (B,m,Di) -> (B,n,Di)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return self.to_out(out)
    # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    # out = rearrange(out, 'b d n -> b n d')
    # return rearrange(self.to_out(out), "b d n -> b n d")
