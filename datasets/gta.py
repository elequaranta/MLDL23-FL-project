from PIL import Image
import numpy as np
import os
from torch import from_numpy
from overrides import override
import torch
from datasets.base_dataset import BaseDataset
import datasets.ss_transforms as tr
from numpy.typing import ArrayLike

from typing import Callable, Dict, List


class GTADataset(BaseDataset):

    class_map = {
        0: 13, # ego_vehicle : vehicle
        1: 0, # road
        2: 1, # sidewalk
        3: 2, # building
        4: 3, # wall
        5: 4, # fence
        6: 5, # pole
        7: 5, # poleGroup: pole
        8: 6, # traffic light
        9: 7, # traffic sign
        10: 8, # vegetation
        11: 9, # terrain
        12: 10, # sky
        13: 11, # person
        14: 12, # rider
        15: 13, # car : vehicle
        16: 13, # truck : vehicle
        17: 13, # bus : vehicle
        18: 14, # motorcycle
        19: 15, # bicycle
    }

    label2train = {
        1: 0, # ego_vehicle : vehicle
        7: 1, # road
        8: 2, # sidewalk
        11: 3, # building
        12: 4, # wall
        13: 5, # fence
        17: 6, # pole
        18: 7, # poleGroup: pole
        19: 8, # traffic light
        20: 9, # traffic sign
        21: 10, # vegetation
        22: 11, # terrain
        23: 12, # sky
        24: 13, # person
        25: 14, # rider
        26: 15, # car : vehicle
        27: 16, # truck : vehicle
        28: 17, # bus : vehicle
        32: 18, # motorcycle
        33: 19, # bicycle
    }
 
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
            os.path.join(self.label_path, f"{sample}.{self.mask_extension}"), 'r')#.convert("L")

    @staticmethod
    @override
    def get_classes_number() -> int:
        return 20
    
    @staticmethod
    @override
    def get_mapping() -> Callable[[torch.Tensor], torch.Tensor]:
        mapping = -1 * np.ones(shape=(256), dtype=np.uint8)
        for k, v in GTADataset.label2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])
    
    @staticmethod
    @override
    def convert_class(class_prediction: ArrayLike) -> ArrayLike:
        out = -1 * np.ones(class_prediction.shape, dtype=np.uint8)
        for id, label in GTADataset.class_map.items():
            out[class_prediction == id] = int(label)
        return out