from abc import ABC, abstractmethod
from typing import Iterator
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel

from config.enums import DatasetOptions
from datasets.idda import IDDADataset
from datasets.gta import GTADataset

class ModelFactory(ABC):

    def __init__(self, dataset_type: DatasetOptions) -> None:
        match dataset_type:
            case DatasetOptions.IDDA:
                self.dataset_class_number = IDDADataset.get_classes_number()
            case DatasetOptions.GTA:
                self.dataset_class_number = GTADataset.get_classes_number()
            case _:
                raise NotImplementedError("The dataset requested is not implemented in ModelFactory")
    
    @abstractmethod
    def construct(self) -> _SimpleSegmentationModel:
        pass

class OptimizerFactory(ABC):

    def __init__(self, lr: float, weight_decay: float, model_params: Iterator[Parameter]) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.params = [{"params": filter(lambda p: p.requires_grad, model_params),
                        'weight_decay': self.weight_decay}]

    @abstractmethod
    def construct(self) -> Optimizer:
        pass

    def update_model_params(self, params_iter: Iterator[Parameter]):
        self.params = [{"params": filter(lambda p: p.requires_grad, params_iter),
                        'weight_decay': self.weight_decay}]

class SchedulerFactory(ABC):

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    @abstractmethod
    def construct(self) -> _LRScheduler:
        pass

    def update_optimizer(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer