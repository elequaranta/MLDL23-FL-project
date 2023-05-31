from abc import ABC, abstractmethod
from typing import List, Optional
from datasets.base_dataset import BaseDataset

import datasets.ss_transforms as sstr

class DatasetFactory(ABC):

    def __init__(self, 
                 root:str, 
                 train_transforms: Optional[sstr.Compose], 
                 test_transforms: Optional[sstr.Compose]) -> None:
        self.root = root
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    @abstractmethod
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        pass

    @abstractmethod
    def construct_test_dataset(self) -> List[BaseDataset]:
        pass