import os
from typing import Callable, Dict, List
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from torch import from_numpy
from overrides import override
import torch
from datasets.base_dataset import BaseDataset
import datasets.ss_transforms as tr

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10] # 21


class IDDADataset(BaseDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None,
                 test_mode: bool = False,
                 client_name: str = None):
        super().__init__(root=root, 
                         transform=transform, 
                         list_samples=list_samples,
                         client_name=client_name,
                         image_extension="jpg",
                         mask_extension="png")
        self.test_mode = test_mode

    @override
    def open_image(self, sample: str) -> Image:
        return Image.open(\
            os.path.join(self.image_path, f"{sample}.{self.image_extension}"), 'r')
    
    @override
    def open_label(self, sample: str) -> Image:
        return Image.open(\
            os.path.join(self.label_path, f"{sample}.{self.mask_extension}"), 'r')
        
    @staticmethod
    @override
    def get_classes_number() -> int:
        return 16

    @staticmethod
    @override
    def get_mapping() -> Callable[[torch.Tensor], torch.Tensor]:
        classes = class_eval
        mapping = np.ones((256,), dtype=np.int64) * -1
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    @staticmethod
    @override
    def convert_class() -> Callable[[torch.Tensor], ArrayLike]:
        return lambda x: x
