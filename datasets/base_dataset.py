import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List
from numpy.typing import ArrayLike
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as sst

class BaseDataset(ABC, VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 image_extension: str,
                 mask_extension: str,
                 transform: sst.Compose = None,
                 client_name: str = None,
                 test_mode: bool = False,):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self._name = client_name
        self.test_mode = test_mode
        self.target_transform = self.get_mapping()
        self.image_extension = image_extension
        self.mask_extension = mask_extension

        self.image_path = os.path.join(root, 'images')
        self.label_path = os.path.join(root, 'labels')

    @staticmethod
    @abstractmethod
    def get_classes_number() -> int:
        pass
    
    @staticmethod
    @abstractmethod
    def get_mapping() -> Callable[[torch.Tensor], torch.Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def convert_class(class_prediction: ArrayLike) -> ArrayLike:
        pass

    def __getitem__(self, index: int) -> Any:
        sample = self.list_samples[index].strip()
        sample = sample.split(".")[0]
        image = self.open_image(sample)
        label = self.open_label(sample)

        if self.transform is not None:
            if not self.test_mode:
                image, label = self.transform(image, label)
            else:
                image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    @abstractmethod
    def open_image(self, sample:str) -> Image:
        pass

    @abstractmethod
    def open_label(self, sample:str) -> Image:
        pass

    def __len__(self) -> int:
        return len(self.list_samples)
    
    @property
    def name(self) -> str:
        return self._name

