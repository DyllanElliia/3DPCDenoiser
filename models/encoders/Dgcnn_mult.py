'''
Author: DyllanElliia
Date: 2025-02-16 19:34:49
LastEditors: DyllanElliia
LastEditTime: 2025-02-16 19:43:16
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DGCNN
from torch_geometric.data import Data


class FeatureExtraction(nn.Module):

  def __init__(self, in_channels, out_channels_list, k_list, dropout=0.0):
    """
        Initialize the DGCNN with multiple convolution layers.
        
        :param in_channels: Input feature dimension
        :param out_channels_list: List of output feature dimensions for each layer
        :param k_list: List of KNN neighbor counts for each layer
        :param dropout: Dropout rate
        """
    super(FeatureExtraction, self).__init__()
    self.in_channels = in_channels
    self.out_channels_list = out_channels_list
    self.k_list = k_list

    self.conv_layers = nn.ModuleList()
    for out_channels, k in zip(out_channels_list, k_list):
      self.conv_layers.append(
          DGCNN(in_channels=in_channels, out_channels=out_channels, k=k))
      in_channels = out_channels  # Update input channels to the output of the previous layer

    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Sequential(
        nn.Linear(in_channels, in_channels //
                  2),  # Final layer output goes through a fully connected layer
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_channels // 2,
                  1)  # Output a value, can be adjusted as needed
    )

  @property
  def out_channels(self):
    return self.out_channels_list[-1]

  def forward(self, x):
    """
        Forward pass.
        
        :param x: Input data, shape (B, N, d)
        :return: Output, shape (B, N, f)
        """
    batch_size, num_points, _ = x.shape
    for i, conv in enumerate(self.conv_layers):
      # Compute the KNN graph for each layer
      edge_index = self.knn_graph(x, self.conv_layers[i].k)
      data = Data(x=x.view(batch_size * num_points, -1), edge_index=edge_index)
      x = conv(data.x, data.edge_index)
      x = F.relu(x)
      x = self.dropout(x)  # Apply dropout after each convolution

    # Final fully connected layer (optional)
    out = self.fc(x.view(batch_size * num_points,
                         -1))  # Flatten before passing to FC
    out = out.view(batch_size, num_points, -1)  # Reshape to (B, N, f)

    return out

  def knn_graph(self, x, k):
    """
        Compute the KNN graph.
        
        :param x: Input point cloud data
        :param k: Number of neighbors
        :return: Edge index of the KNN graph
        """
    from torch_geometric.nn import knn_graph
    edge_index = knn_graph(x.view(-1, x.size(-1)), k=k, batch=None)
    return edge_index
