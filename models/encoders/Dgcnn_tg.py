'''
Author: DyllanElliia
Date: 2025-02-16 18:59:28
LastEditors: DyllanElliia
LastEditTime: 2025-02-16 19:49:15
Description: 
'''

import torch
from torch.nn import Module, Linear, ModuleList
from torch.nn import Sequential as Seq, Linear, BatchNorm1d as BN, ReLU
import torch.nn as nn
import pytorch3d.ops
from ..utils import *

from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.models import EdgeCNN

np.longlong(9**19)


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
    # self.dynamic_graph = dynamic_graph
    self.num_convs = num_convs
    # self.mlp_feat = mlp_feat
    # self.cat_feat = cat_feat

    # Edge Convolution Units
    self.convs = ModuleList()
    # if self.mlp_feat == True:
    #     self.lin = ModuleList()
    in_channels = 3
    for out_channels in [32, 32, 64, 64, 128, 256, 128, 128]:
      conv = DynamicEdgeConv(
          EdgeCNN(
              in_channels=in_channels,
              hidden_channels=out_channels,
              out_channels=out_channels,
              num_layers=conv_num_fc_layers,
              act=activation,
              jk=conv_aggr,
          ),
          k=conv_knn,
      )

      self.convs.append(conv)
      # if self.mlp_feat:
      #     mlp = Seq(
      #         Linear(in_channels, conv.out_channels),
      #         BN(conv.out_channels),
      #         ReLU(),
      #     )
      #     self.lin.append(mlp)
      in_channels = out_channels
      # self.dropout = nn.Dropout(p=0.1)

  @property
  def out_channels(self):
    return 128

  def forward(self, x):
    # x: [B, N, 3]
    # -> feat: [B, N, F]
    feat = x
    for net in self.convs:
      feat = net(feat)
    return feat
