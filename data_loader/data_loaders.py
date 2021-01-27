import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from data_loader.dtu_yao import MVSDataset

np.random.seed(1234)


class MVSLoader(DataLoader):

    def __init__(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, depth_scale=1.0):
        self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                      shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, depth_scale=depth_scale)
        sampler = SequentialSampler(self.mvs_dataset)
        super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                         num_workers=4, pin_memory=True)

        self.n_samples = len(self.mvs_dataset)
        self.device = None

    def shuffle_and_crop(self):
        self.mvs_dataset.generate_indices()

    def set_device(self, device):
        self.device = device

    def get_num_samples(self):
        return len(self.mvs_dataset)


