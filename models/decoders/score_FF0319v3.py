import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlockConv1d(nn.Module):
  """1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

  def __init__(
      self,
      c_dim,
      size_in,
      size_h=None,
      size_out=None,
      norm_method="batch_norm",
      legacy=False,
  ):
    super().__init__()
    # Attributes
    if size_h is None:
      size_h = size_in
    if size_out is None:
      size_out = size_in

    self.size_in = size_in
    self.size_h = size_h
    self.size_out = size_out
    # Submodules
    if norm_method == "batch_norm":
      norm = nn.BatchNorm1d
    elif norm_method == "sync_batch_norm":
      norm = nn.SyncBatchNorm
    else:
      raise Exception("Invalid norm method: %s" % norm_method)

    self.bn_0 = norm(size_in)
    self.bn_1 = norm(size_h)

    self.fc_0 = nn.Conv1d(size_in, size_h, 1)
    self.fc_1 = nn.Conv1d(size_h, size_out, 1)
    self.fc_c = nn.Conv1d(c_dim, size_out, 1)
    self.actvn = nn.ReLU()

    if size_in == size_out:
      self.shortcut = None
    else:
      self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

    # Initialization
    nn.init.zeros_(self.fc_1.weight)
    # self.dropout = nn.Dropout(p=0.05)

  def forward(self, x, c):
    net = self.fc_0(self.actvn(self.bn_0(x)))
    # net = self.dropout(net)
    dx = self.fc_1(self.actvn(self.bn_1(net)))

    if self.shortcut is not None:
      x_s = self.shortcut(x)
    else:
      x_s = x

    out = x_s + dx + self.fc_c(c)

    return out


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
      hidden_h=None,
      decoder_h=None,
  ):
    """
        Args:
            z_dim:   Dimension of context vectors.
            dim:     Point dimension.
            out_dim: Gradient dim.
            hidden_size:   Hidden states dim.
        """
    super().__init__()
    if hidden_h == None:
      self.hidden_h = hidden_size
    else:
      self.hidden_h = hidden_h
    self.z_dim = z_dim
    self.dim = dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_blocks = num_blocks

    # Input = Conditional = zdim (code) + dim (xyz)
    self.pos_e_dim = 64
    self.pos_embedding = nn.Linear(dim, self.pos_e_dim)
    c_dim = z_dim + self.pos_e_dim
    self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
    self.blocks = nn.ModuleList([
        ResnetBlockConv1d(c_dim, hidden_size, self.hidden_h)
        for _ in range(num_blocks)
    ])

    self.decoder_h = decoder_h
    if self.decoder_h:
      print("have decoder", self.decoder_h)
      self.bn_64 = nn.BatchNorm1d(hidden_size)
      self.conv_64 = nn.Conv1d(hidden_size, self.decoder_h, 1)
      self.actvn_64 = nn.ReLU()

      self.bn_out = nn.BatchNorm1d(self.decoder_h)
      self.conv_out = nn.Conv1d(self.decoder_h, out_dim, 1)
      self.actvn_out = nn.ReLU()
    else:
      self.bn_out = nn.BatchNorm1d(hidden_size)
      self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
      self.actvn_out = nn.ReLU()

    self.fusion_weight = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.ReLU(),
                                       nn.Conv1d(hidden_size, 1, 1),
                                       nn.Sigmoid())

  def forward(self, x, c):
    """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim) Shape latent code
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """

    batch_size, num_points, _ = x.size()
    p = self.pos_embedding(x).transpose(1, 2)  # (bs, dim, n_points)
    c_expand = c.unsqueeze(2).expand(-1, -1, num_points)
    c_xyz = torch.cat([p, c_expand], dim=1)
    net = self.conv_p(c_xyz)
    for block in self.blocks:
      net = block(net, c_xyz)
    w = self.fusion_weight(net).transpose(1, 2)  # (B, n, 1)
    if self.decoder_h:
      net = self.conv_64(self.actvn_64(self.bn_64(net)))
    out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
    return out, w
