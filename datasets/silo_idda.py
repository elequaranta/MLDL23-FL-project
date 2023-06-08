from typing import List

import torch
from datasets.idda import IDDADataset
import datasets.ss_transforms as tr


class SiloIddaDataset(IDDADataset):

    def __init__(self, 
                 root: str, 
                 list_samples: List[str], 
                 transform: tr.Compose = None, 
                 test_mode: bool = False, 
                 client_name: str = None):
        super().__init__(root, 
                         list_samples, 
                         transform, 
                         test_mode, 
                         client_name)
        self.mean, self.std = self.get_mean_std()

    def get_mean_std(self):
        psum = torch.zeros(size=(3,))
        psum_sq = torch.zeros(size=(3,))
        for sample in self.list_samples:
            sample = sample.strip().split(".")[0]
            img = self.open_image(sample)
            psum += img.sum( axis = [1, 2])
            psum_sq += (img ** 2).sum(axis = [1, 2])
        count = (len(self.list_samples) - 1) * 1920 * 1080
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)
        return total_mean, total_std