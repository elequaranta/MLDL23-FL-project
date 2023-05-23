import os
from typing import Any, List
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None,
                 test_mode: bool = False,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.test_mode = test_mode
        self.target_transform = self.get_mapping()

        self.image_path = os.path.join(root, 'images')
        self.label_path = os.path.join(root, 'labels')

    @staticmethod
    def get_classes_number() -> int:
        return 16

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        sample = self.list_samples[index].strip()
        image = Image.open(os.path.join(self.image_path, f"{sample}.jpg"), 'r')
        label = Image.open(os.path.join(self.label_path, f"{sample}.png"), 'r')

        if self.transform is not None:
            if not self.test_mode:
                image, label = self.transform(image, label)
            else:
                image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
