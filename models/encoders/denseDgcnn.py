import torch
from torch.nn import Module, Linear, ModuleList
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU
import torch.nn as nn
import pytorch3d.ops
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


class DenseEdgeConv(Module):

  def __init__(
      self,
      in_channels,
      num_fc_layers,
      growth_rate,
      knn=16,
      aggr="max",
      activation="relu",
      relative_feat_only=False,
      mlp_feat=False,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.knn = knn
    assert num_fc_layers > 2
    self.num_fc_layers = num_fc_layers
    self.growth_rate = growth_rate
    self.relative_feat_only = relative_feat_only
    self.mlp_feat = mlp_feat

    # Densely Connected Layers
    if relative_feat_only:
      self.layer_first = FCLayer(in_channels,
                                 growth_rate,
                                 bias=True,
                                 activation=activation)
    else:
      self.layer_first = FCLayer(3 * in_channels,
                                 growth_rate,
                                 bias=True,
                                 activation=activation)
    self.layer_last = FCLayer(
        in_channels + (num_fc_layers - 1) * growth_rate,
        growth_rate,
        bias=True,
        activation=None,
    )
    if mlp_feat:
      self.lin_first = Seq(
          Linear(
              in_channels,
              in_channels + growth_rate,
          ),
          BN(in_channels + growth_rate),
          ReLU(),
      )

      self.lin_last = Seq(
          Linear(
              in_channels + (num_fc_layers - 1) * growth_rate,
              in_channels + num_fc_layers * growth_rate,
          ),
          BN(in_channels + num_fc_layers * growth_rate),
          ReLU(),
      )
    self.layers = ModuleList()

    for i in range(1, num_fc_layers - 1):
      self.layers.append(
          FCLayer(
              in_channels + i * growth_rate,
              growth_rate,
              bias=True,
              activation=activation,
          ))
    if mlp_feat:
      self.lins = ModuleList()
      for i in range(1, num_fc_layers - 1):
        # # v1
        # self.lins.append(
        #     Seq(
        #         Linear(in_channels + i * growth_rate, growth_rate),
        #         BN(growth_rate),
        #         ReLU(),
        #     )
        # )
        # v2
        self.lins.append(
            Seq(
                Linear(
                    in_channels + i * growth_rate,
                    in_channels + (i + 1) * growth_rate,
                ),
                BN(in_channels + (i + 1) * growth_rate),
                ReLU(),
            ))
    # if mlp_feat:
    #     out_channels = self.in_channels + self.num_fc_layers * self.growth_rate
    #     self.lin = Seq(
    #         Linear(in_channels, out_channels),
    #         BN(out_channels),
    #         ReLU(),
    #     )

    self.aggr = Aggregator(aggr)

  @property
  def out_channels(self):
    return self.in_channels + self.num_fc_layers * self.growth_rate

  def get_edge_feature(self, x, knn_idx):
    """
        :param  x:          (B, N, d)
        :param  knn_idx:    (B, N, K)
        :return (B, N, K, 2*d)
        """
    knn_feat = knn_group(x, knn_idx)  # B * N * K * d
    x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
    if self.relative_feat_only:
      edge_feat = knn_feat - x_tiled
    else:
      edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
    return edge_feat

  def forward(self, x, pos):
    """
        :param  x:  (B, N, d)
        :return (B, N, d+L*c)
        """
    knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)
    B = x.size(0)
    N = x.size(1)
    K = self.knn

    # First Layer
    edge_feat = self.get_edge_feature(x, knn_idx)

    y = torch.cat(
        [
            self.layer_first(edge_feat),  # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1),  # (B, N, K, d)
        ],
        dim=-1,
    )  # (B, N, K, d+c)
    if self.mlp_feat:
      x1 = self.lin_first(
          x.unsqueeze(-2).repeat(1, 1, self.knn,
                                 1).view(B * N * K,
                                         -1)).view(B, N, K,
                                                   -1)  # (B, N, K, c+d)
      y = y + x1

    # Intermediate Layers
    # for layer in self.layers:
    for i in range(0, self.num_fc_layers - 2):
      # v1
      # y1 = self.layers[i](y)  # (B, N, K, c)
      # if self.mlp_feat:
      #     # print("useMlp")
      #     y2 = self.lins[i](y.view(B * N * K, -1)).view(B, N, K, -1)
      #     # print(y1.shape, y2.shape)
      #     y1 = y1 + y2
      # y = torch.cat(
      #     [
      #         y1,
      #         y,  # (B, N, K, c+d)
      #     ],
      #     dim=-1,
      # )  # (B, N, K, d+c+...)
      y1 = self.layers[i](y)  # (B, N, K, c)
      if self.mlp_feat:
        # print("useMlp")
        y2 = self.lins[i](y.view(B * N * K, -1)).view(B, N, K, -1)
        # print(y1.shape, y2.shape)
        # y1 = y1 + y2
        y = (
            torch.cat(
                [
                    y1,
                    y,  # (B, N, K, c+d)
                ],
                dim=-1,
            ) + y2)  # (B, N, K, d+c+...)
      else:
        y = torch.cat(
            [
                y1,
                y,  # (B, N, K, c+d)
            ],
            dim=-1,
        )  # (B, N, K, d+c+...)
    if self.mlp_feat:
      # print(y.shape)
      xl = self.lin_last(y.view(B * N * K, -1)).view(B, N, K, -1)
      # Last Layer
      y = (
          torch.cat(
              [self.layer_last(y), y],
              dim=-1,  # (B, N, K, c)  # (B, N, K, d+(L-1)*c)
          ) + xl)  # (B, N, K, d+L*c)
    else:
      # Last Layer
      y = torch.cat(
          [self.layer_last(y), y],
          dim=-1  # (B, N, K, c)  # (B, N, K, d+(L-1)*c)
      )  # (B, N, K, d+L*c)

    # Pooling
    y = self.aggr(y, dim=-2)

    return y


class FeatureExtraction(Module):

  def __init__(
      self,
      in_channels=3,
      dynamic_graph=True,
      conv_channels=24,
      num_convs=4,
      conv_num_fc_layers=3,
      conv_growth_rate=12,
      conv_knn=16,
      conv_aggr="max",
      activation="relu",
      mlp_feat=False,
      cat_feat=False,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.dynamic_graph = dynamic_graph
    self.num_convs = num_convs
    self.mlp_feat = mlp_feat
    self.cat_feat = cat_feat

    # Edge Convolution Units
    self.transforms = ModuleList()
    self.convs = ModuleList()
    # if self.mlp_feat == True:
    #     self.lin = ModuleList()
    for i in range(num_convs):
      if i == 0:
        trans = FCLayer(in_channels, conv_channels, bias=True, activation=None)
        conv = DenseEdgeConv(
            conv_channels,
            num_fc_layers=conv_num_fc_layers,
            growth_rate=conv_growth_rate,
            knn=conv_knn,
            aggr=conv_aggr,
            activation=activation,
            relative_feat_only=True,
            mlp_feat=self.mlp_feat,
        )
      else:
        trans = FCLayer(in_channels,
                        conv_channels,
                        bias=True,
                        activation=activation)
        conv = DenseEdgeConv(
            conv_channels,
            num_fc_layers=conv_num_fc_layers,
            growth_rate=conv_growth_rate,
            knn=conv_knn,
            aggr=conv_aggr,
            activation=activation,
            relative_feat_only=False,
            mlp_feat=self.mlp_feat,
        )
      self.transforms.append(trans)
      self.convs.append(conv)
      # if self.mlp_feat:
      #     mlp = Seq(
      #         Linear(in_channels, conv.out_channels),
      #         BN(conv.out_channels),
      #         ReLU(),
      #     )
      #     self.lin.append(mlp)
      in_channels = conv.out_channels
      # self.dropout = nn.Dropout(p=0.1)
      self._requires_grad_ = True

  @property
  def out_channels(self):
    if self.cat_feat:
      return self.convs[-1].out_channels * self.num_convs
    else:
      return self.convs[-1].out_channels

  def dynamic_graph_forward(self, x):
    for i in range(self.num_convs):
      x = self.transforms[i](x)
      x = self.convs[i](x, x)
      if self.cat_feat:
        if i == 0:
          x_cat = x.clone()
        else:
          x_cat = torch.cat([x, x_cat], dim=2)
    if self.cat_feat:
      return x_cat
    else:
      return x

  # def dynamic_graph_forward_mlp(self, x):
  #     B = x.size(0)
  #     N = x.size(1)
  #     print("useMlp")
  #     for i in range(self.num_convs):
  #         y = self.lin[i](x.view(B * N, -1)).view(B, N, -1)
  #         x = self.transforms[i](x)
  #         x = self.convs[i](x, x)
  #         x = x + y
  #     return x

  def static_graph_forward(self, pos):
    x = pos
    for i in range(self.num_convs):
      x = self.transforms[i](x)
      x = self.convs[i](x, pos)
    return x

  def forward(self, x):
    if self._requires_grad_:
      if self.dynamic_graph:
        return self.dynamic_graph_forward(x)
      else:
        return self.static_graph_forward(x)
    else:
      with torch.no_grad():
        if self.dynamic_graph:
          return self.dynamic_graph_forward(x)
        else:
          return self.static_graph_forward(x)
