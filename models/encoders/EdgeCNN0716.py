import torch
from torch.nn import Module, Linear, ModuleList
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU
import torch.nn as nn
import pytorch3d.ops
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import reset
from ..utils import *


def get_knn_idx(x, y, k, offset=0):
  """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
  _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k + offset)
  return knn_idx[:, :, offset:]


def knn_group(x: torch.FloatTensor, idx: torch.LongTensor):
  """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
  B, N, F = tuple(x.size())
  _, M, k = tuple(idx.size())

  x = x.unsqueeze(1).expand(B, M, N, F)
  idx = idx.unsqueeze(3).expand(B, M, k, F)

  return torch.gather(x, dim=2, index=idx)


class Aggregator(torch.nn.Module):

  def __init__(self, oper):
    super().__init__()
    assert oper in ("mean", "sum", "max")
    self.oper = oper

  def forward(self, x, dim=2):
    if self.oper == "mean":
      return x.mean(dim=dim, keepdim=False)
    elif self.oper == "sum":
      return x.sum(dim=dim, keepdim=False)
    elif self.oper == "max":
      ret, _ = x.max(dim=dim, keepdim=False)
      return ret


class EdgeConv(MessagePassing):

  def __init__(self, in_channels, out_channels):
    super().__init__(aggr='max')  #  "Max" aggregation.
    self.mlp = Seq(Linear(2 * in_channels, out_channels), BN(out_channels),
                   ReLU(), Linear(out_channels, out_channels), BN(out_channels),
                   ReLU())
    self.lin = Seq(Linear(in_channels, out_channels), BN(out_channels), ReLU())

  # def forward(self, x, edge_index):
  #   # x has shape [N, in_channels]
  #   # edge_index has shape [2, E]

  #   x_pair = (x, x)
  #   out_1 = self.propagate(edge_index, x=x_pair[0])
  #   out_2 = self.lin(x_pair[1])

  #   return out_1 + out_2

  def forward(self, x, edge_index):
    # x has shape [N, in_channels]
    # edge_index has shape [2, E]

    x_pair = (x, x)
    out_1 = self.propagate(edge_index, x=x_pair[0])
    out_2 = self.lin(x_pair[1])

    return out_1 + out_2

  def message(self, x_i, x_j):
    # x_i has shape [E, in_channels]
    # x_j has shape [E, in_channels]

    tmp = torch.cat([x_i, x_j - x_i],
                    dim=1)  # tmp has shape [E, 2 * in_channels]
    return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):

  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels)

  def forward(self, x, edge_index):
    return super().forward(x, edge_index)


class FeatureExtraction(Module):

  def __init__(self, k=32, input_dim=0, embedding_dim=512):
    super(FeatureExtraction, self).__init__()
    self.k = k
    self.input_dim = input_dim
    # self.z_dim = z_dim
    self.embedding_dim = embedding_dim
    # self.output_dim = output_dim

    self.conv1 = DynamicEdgeConv(3, 16)
    self.conv2 = DynamicEdgeConv(16, 48)
    self.conv3 = DynamicEdgeConv(48, 64)
    self.conv4 = DynamicEdgeConv(64, 128)
    self.conv5 = DynamicEdgeConv(16 + 48 + 64 + 128, self.embedding_dim)

    # self.linear1 = nn.Linear(self.embedding_dim, 256, bias=False)
    # self.linear2 = nn.Linear(256, 128)
    # self.linear3 = nn.Linear(128, self.output_dim)

    # if self.z_dim > 0:
    #   self.linear_proj = nn.Linear(512, self.z_dim)
    #   self.dropout_proj = nn.Dropout(0.1)

    self.reset_parameters()

  def reset_parameters(self):
    reset(self.conv1)
    reset(self.conv2)
    reset(self.conv3)
    reset(self.conv4)
    # reset(self.linear1)
    # reset(self.linear2)
    # reset(self.linear3)

  @property
  def out_channels(self):
    return self.embedding_dim

  def get_edge_index(self, x):
    cols = get_knn_idx(x, x, self.k + 1).view(self.batch_size, self.num_points,
                                              -1)
    cols = (cols + self.rows_add).view(1, -1)
    edge_index = torch.cat([cols, self.rows], dim=0)
    edge_index, _ = remove_self_loops(edge_index.long())

    return edge_index

  def forward(self, x):
    self.batch_size = x.size(0)
    self.num_points = x.size(1)

    self.rows = torch.arange(0,
                             self.num_points).unsqueeze(0).unsqueeze(2).repeat(
                                 self.batch_size, 1, self.k + 1).cuda()
    self.rows_add = (
        self.num_points *
        torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(
            1, self.num_points, self.k + 1).cuda()
    self.rows = (self.rows + self.rows_add).view(1, -1)

    # if disp_feat is not None:
    #   disp_feat = F.relu(self.linear_proj(disp_feat))
    #   disp_feat = self.dropout_proj(disp_feat)
    #   x = torch.cat([x, disp_feat], dim=-1)

    edge_index = self.get_edge_index(x)
    x = x.view(self.batch_size * self.num_points, -1)
    x1 = self.conv1(x, edge_index)
    x1 = x1.view(self.batch_size, self.num_points, -1)

    edge_index = self.get_edge_index(x1)
    x1 = x1.view(self.batch_size * self.num_points, -1)
    x2 = self.conv2(x1, edge_index)
    x2 = x2.view(self.batch_size, self.num_points, -1)

    edge_index = self.get_edge_index(x2)
    x2 = x2.view(self.batch_size * self.num_points, -1)
    x3 = self.conv3(x2, edge_index)
    x3 = x3.view(self.batch_size, self.num_points, -1)

    edge_index = self.get_edge_index(x3)
    x3 = x3.view(self.batch_size * self.num_points, -1)
    x4 = self.conv4(x3, edge_index)
    x4 = x4.view(self.batch_size, self.num_points, -1)

    edge_index = self.get_edge_index(x4)
    x4 = x4.view(self.batch_size * self.num_points, -1)
    x_combined = torch.cat((x1, x2, x3, x4), dim=-1)
    x_combined = x_combined.view(self.batch_size * self.num_points, -1)
    x = self.conv5(x_combined, edge_index)
    x = x.view(self.batch_size, self.num_points, -1)

    return x
