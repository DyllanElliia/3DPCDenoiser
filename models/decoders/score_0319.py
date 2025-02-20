import torch
import torch.nn as nn
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from math import ceil, pi, sqrt


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
    def __init__(self, dim, dim_out=None, dropout=0.0):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = PixelNorm(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.0)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.0)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, *, dropout=0.0):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = (
            nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, mask=None):
        res = self.residual_conv(x)
        h = self.block1(x, mask=mask)
        h = self.block2(h, mask=mask)
        h = self.excite(h, mask=mask)
        return h + res


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(384, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc1_1 = nn.Linear(512, 256)
        self.fc1_2 = nn.Linear(256, 64)
        self.fc1_3 = nn.Linear(64, 3)
        self.bn1_11 = nn.BatchNorm1d(256)
        self.bn1_22 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = F.relu(self.bn1_11(self.fc1_1(x)))
        x = F.relu(self.bn1_22(self.fc1_2(x)))
        x = torch.tanh(self.fc1_3(x))
        return x


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

        # Input = Conditional = zdim (code) + dim (xyz)
        c_dim = z_dim + dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [ResnetBlock(hidden_size, hidden_size) for _ in range(num_blocks)]
        )

        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn_out = nn.ReLU()

    def forward(self, x, c):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim) Shape latent code
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        p = x.transpose(1, 2)  # (bs, dim, n_points)
        batch_size, D, num_points = p.size()

        c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
        c_xyz = torch.cat([p, c_expand], dim=1)
        net = self.conv_p(c_xyz)
        for block in self.blocks:
            net = block(net)

        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out
