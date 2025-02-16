import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class PointCloudDataset(Dataset):

  def __init__(self,
               root,
               dataset,
               split,
               resolution,
               transform=None,
               pc='pointclouds'):
    super().__init__()
    self.pcl_dir = os.path.join(root, dataset, pc, split, resolution)
    self.transform = transform
    self.pointclouds = []
    self.pointcloud_names = []
    file_list = sorted(os.listdir(self.pcl_dir))
    print('Loading %d point clouds from %s' % (len(file_list), self.pcl_dir))
    print('First 5 files:', file_list[:5])
    for fn in tqdm(file_list, desc='Loading'):
      if fn[-3:] != 'xyz':
        continue
      pcl_path = os.path.join(self.pcl_dir, fn)
      if not os.path.exists(pcl_path):
        raise FileNotFoundError('File not found: %s' % pcl_path)
      pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
      self.pointclouds.append(pcl)
      self.pointcloud_names.append(fn[:-4])

  def __len__(self):
    return len(self.pointclouds)

  def __getitem__(self, idx):
    data = {
        'pcl_clean': self.pointclouds[idx].clone(),
        'name': self.pointcloud_names[idx]
    }
    if self.transform is not None:
      data = self.transform(data)
    return data
