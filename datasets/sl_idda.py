import numpy as np
from numpy.typing import ArrayLike
import copy
from typing import List

from overrides import override
from torch import Tensor
import torch
from datasets.gta import GTADataset
from datasets.idda import IDDADataset
import datasets.ss_transforms as tr


class IDDADatasetSelfLearning(IDDADataset):

    def __init__(self, 
                 root: str, 
                 list_samples: List[str], 
                 transform: tr.Compose = None, 
                 test_mode: bool = False, 
                 client_name: str = None):
        super().__init__(root, list_samples, transform, test_mode, client_name)
        self.labels = []

    @override
    def __getitem__(self, index: int):
        sample = self.list_samples[index].strip()
        sample = sample.split(".")[0]
        image = self.open_image(sample)
        
        if self.transform is not None:
            image = self.transform(image)

        if len(self.labels) == len(self.list_samples):
            return image, self.labels[index]

        return image, torch.Tensor([-1])
    
    @staticmethod
    @override
    def get_classes_number() -> int:
        return 16
 
    @staticmethod
    @override
    def convert_class(class_prediction: ArrayLike) -> ArrayLike:
        out = -1 * np.ones(class_prediction.shape, dtype=np.uint8)
        for id, label in GTADataset.class_map.items():
            out[class_prediction == id] = int(label)
        return out
    
    def update_labels(self, labels: List[Tensor]) -> None:
        # TODO: check if this deepcopy is needed
        self.labels = copy.deepcopy(labels)