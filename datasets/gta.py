import os
from PIL import Image
import numpy as np
from torch import from_numpy
from overrides import override
import torch
from datasets.base_dataset import BaseDataset
import datasets.ss_transforms as tr
from numpy.typing import ArrayLike

from typing import Callable, Dict, List


class GTADataset(BaseDataset):

    # Some of the RGB value of every class of the palette (the sum are unique)
    PALETTE = np.array([320, #0
                        511, #1
                        210, #2
                        360, #3
                        496, #4
                        459, #5
                        450, #6
                        440, #7
                        284, #8
                        555, #9
                        380, #10
                        300, #11
                        255, #12
                        142, #13
                        70,  #14
                        160, #15
                        180, #16
                        230, #17
                        162  #18
                        ], dtype=np.int32)

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, 
                         transform=transform, 
                         list_samples=list_samples,
                         client_name=client_name,
                         image_extension="png",
                         mask_extension="png")

    @override
    def open_image(self, sample: str) -> Image:
        return Image.open(\
            os.path.join(self.image_path, f"{sample}.{self.image_extension}"), 'r').convert("RGB")
    
    @override
    def open_label(self, sample: str) -> Image:
        return Image.open(\
            os.path.join(self.label_path, f"{sample}.{self.mask_extension}"), 'r').convert("RGB")

    @staticmethod
    @override
    def get_classes_number() -> int:
        return 19
    
    @staticmethod
    @override
    def get_mapping() -> Callable[[torch.Tensor], torch.Tensor]:
        mapping = np.ones((765,), dtype=np.int64) * -1
        for i, v in enumerate(GTADataset.PALETTE):
            mapping[v] = i
        #return lambda x: from_numpy(mapping[x])
        return lambda x: from_numpy(mapping[x.sum(dim=2)])
    
    @staticmethod
    @override
    def convert_class() -> Callable[[torch.Tensor], ArrayLike]:
        # map class from gta info.json + class map ida
        #map_class = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,13,13,14,15])
        # map class from project instruction
        map_class = np.array([13,0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,13,14,15])
        return lambda x: map_class[x]