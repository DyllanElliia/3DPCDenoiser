import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClamp(nn.Module):

  def __init__(self, min_val=0.0, max_val=1.0):
    super(SoftClamp, self).__init__()
    self.min_val = min_val
    self.max_val = max_val
    self.scale = max_val - min_val

  def forward(self, x):
    # return self.min_val + self.scale * (F.softplus(x) / (F.softplus(x) + 1))
    return F.softplus(-(F.softplus(x) - self.scale)) + self.min_val
