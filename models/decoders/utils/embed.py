'''
Author: DyllanElliia
Date: 2024-05-07 13:56:19
LastEditors: DyllanElliia
LastEditTime: 2024-05-07 13:56:20
Description: 
'''
from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat

from torch_cluster import fps

from .define import default, exists, cache_fn


class PointEmbed(nn.Module):

  def __init__(self, hidden_dim=48, dim=128):
    super().__init__()

    assert hidden_dim % 6 == 0

    self.embedding_dim = hidden_dim
    e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
    e = torch.stack([
        torch.cat([
            e,
            torch.zeros(self.embedding_dim // 6),
            torch.zeros(self.embedding_dim // 6)
        ]),
        torch.cat([
            torch.zeros(self.embedding_dim // 6), e,
            torch.zeros(self.embedding_dim // 6)
        ]),
        torch.cat([
            torch.zeros(self.embedding_dim // 6),
            torch.zeros(self.embedding_dim // 6), e
        ]),
    ])
    self.register_buffer('basis', e)  # 3 x 16

    self.mlp = nn.Conv1d(self.embedding_dim + 3, dim, 1)

  @staticmethod
  def embed(input, basis):
    # (B,N,D) x (D,E) -> (B,N,E)
    projections = torch.einsum('bnd,de->bne', input, basis)
    # (B,N,E) -> (B,N, 2*E: cat[sin(E),cos(E)])
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings

  def forward(self, input):
    # input: B x N x 3
    # (B,N,3) -> (B,N,2E+3) -> (B,N,C)
    embed = self.mlp(
        rearrange(torch.cat([self.embed(input, self.basis), input], dim=2),
                  "B N E -> B E N"))  # B x N x C
    return rearrange(embed, "B C N -> B N C")


def nonlinearity(x):
  # swish
  return x * torch.sigmoid(x)


class TimeStepEmbedding(nn.Module):

  def __init__(self, ch: int):
    super().__init__()
    self.ch = ch
    # timestep embedding
    self.dense = nn.ModuleList([
        torch.nn.Conv1d(self.ch, self.ch, 1),
        # torch.nn.Conv1d(self.ch, self.ch, 1),
    ])

  def forward(self, t: torch.Tensor):
    # input: (B)
    # output: (B,ch)
    # timestep embedding
    temb = rearrange(self.get_timestep_embedding(t, self.ch), "B C -> B C ()")
    # print(temb.shape)
    temb = self.dense[0](temb)
    # print(temb.shape)
    temb = nonlinearity(temb)
    # print(temb.shape)
    # temb = self.dense[1](temb)
    # print(temb.shape)
    return rearrange(temb, "B C 1 -> B C")

  def __len__(self):
    return self.ch

  def get_timestep_embedding(self, timesteps, embedding_dim):
    """
      This matches the implementation in Denoising Diffusion Probabilistic Models:
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
      emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeStepEmbedder_2(nn.Module):

  def __init__(self, ch: int, dim: int):
    super().__init__()
    self.dim = dim
    self.ch = ch
    # timestep embedding
    self.mlp = nn.Sequential(
        torch.nn.Linear(self.dim, self.ch, bias=False),
        nn.ReLU(inplace=True),
        torch.nn.Linear(self.ch, self.ch, bias=False),
        # torch.nn.Conv1d(self.ch, self.ch, 1),
    )

  def forward(self, t: torch.Tensor):
    # input: (B)
    # output: (B,ch)
    # timestep embedding
    temb = self.timestep_embedding(t, self.dim)
    # print(temb.shape)
    # temb = self.dense[0](temb)
    # print(temb.shape)
    # temb = nonlinearity(temb)
    # print(temb.shape)
    # temb = self.dense[1](temb)
    # print(temb.shape)
    return self.mlp(temb)

  def __len__(self):
    return self.ch

  def timestep_embedding(self, t, dim, max_period=10000):
    """
          Create sinusoidal timestep embeddings.
          :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
          :param dim: the dimension of the output.
          :param max_period: controls the minimum frequency of the embeddings.
          :return: an (N, D) Tensor of positional embeddings.
          """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=t.device)
    args = t[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
          [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
