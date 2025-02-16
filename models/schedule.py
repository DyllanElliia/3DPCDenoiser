'''
Author: DyllanElliia
Date: 2024-05-27 15:24:23
LastEditors: DyllanElliia
LastEditTime: 2024-05-27 15:24:23
Description: 
'''
import torch
import torch.nn.functional as F
from typing import Union
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
import time


class VarianceSchedule(Module):

  def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
    super().__init__()
    assert mode in ("linear",)
    self.num_steps = num_steps
    self.beta_1 = beta_1
    self.beta_T = beta_T
    self.mode = mode

    if mode == "linear":
      betas = torch.linspace(0, beta_T, steps=num_steps + 1)
    # self.betas = betas
    # betas = self.betas[1:]
    # betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

    alphas = 1 - betas
    log_alphas = torch.log(alphas)
    for i in range(1, log_alphas.size(0)):  # 1 to T
      log_alphas[i] += log_alphas[i - 1]
    alpha_bars = log_alphas.exp()

    sigma_bars = torch.zeros_like(betas)
    sigma_bars = (1 - alpha_bars) / alpha_bars

    sigmas_flex = betas
    for i in range(1, sigmas_flex.size(0)):
      sigmas_flex[i] /= alpha_bars[i - 1]

    sigmas_inflex = torch.zeros_like(sigmas_flex)
    for i in range(1, sigmas_flex.size(0)):
      sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) /
                          ((1 - alpha_bars[i]) * alpha_bars[i - 1]) * betas[i])
    sigmas_flex = torch.sqrt(sigmas_flex)
    sigmas_inflex = torch.sqrt(sigmas_inflex)
    sigmas_flex = torch.nan_to_num(sigmas_flex, nan=0.0)
    sigmas_inflex = torch.nan_to_num(sigmas_inflex, nan=0.0)

    self.register_buffer("betas", betas)
    self.register_buffer("alphas", alphas)
    self.register_buffer("alpha_bars", alpha_bars)
    self.register_buffer("sigmas_flex", sigmas_flex)
    self.register_buffer("sigmas_inflex", sigmas_inflex)
    # self.register_buffer("sigma_bars", sigma_bars)

  def uniform_sample_t(self, batch_size, max=-1):
    ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
    return ts.tolist()

  # use it for training
  def uniform_timestep(self, batch_size, t_min=1, t_max=-1):
    ts = np.random.choice(
        np.arange(t_min, self.num_steps + 1 if t_max == -1 else t_max),
        batch_size)
    return ts.tolist()

  def eqdis_uniform_sample_t(self, batch_size, max=-1):
    return (np.random.permutation((
        (np.arange(0, batch_size) / batch_size + np.random.uniform(0.0, 1.0)) %
        1.0) * self.num_steps + 1).astype(int).tolist())

  def get_sigmas(self, t, flexibility):
    assert 0 <= flexibility and flexibility <= 1
    sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
        1 - flexibility)
    return sigmas
    # return 0.0
  def search_t(self, sigma):
    # binary search
    left = 0
    right = self.num_steps
    sigma2 = sigma**2
    while left < right:
      mid = (left + right) // 2
      if self.get_sigma_bar(mid) < sigma2:
        left = mid + 1
      else:
        right = mid
    return left

  # use it for calculating sigma
  def get_sigma_bar(self, t):
    return (1 - self.alpha_bars[t]) / self.alpha_bars[t]

  def vs_sigma_acc(self):
    cnt = 0
    for t in range(self.num_steps + 1):
      alpha_bar_t = self.alpha_bars[t]
      cnt += torch.sqrt((1 - alpha_bar_t) / (alpha_bar_t))
    return cnt

  def vs_sigma(self):
    # TODO: first is much more right, but result is bad
    alpha_bar_t = self.alpha_bars[self.num_steps]
    return torch.sqrt((1 - alpha_bar_t) / (alpha_bar_t))
    return self.vs_sigma_acc()
    # return (1 - alpha_bar_t) / (alpha_bar_t)
    # cnt = 0
    # for t in range(self.num_steps + 1):
    #     alpha_bar_t = self.alpha_bars[t]
    #     cnt += torch.sqrt((1 - alpha_bar_t) / (alpha_bar_t))
    # return cnt
