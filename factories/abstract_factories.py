from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, Iterator, List
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel
from torchvision.datasets import VisionDataset
from datasets.base_dataset import BaseDataset
from datasets.gta import GTADataset

import datasets.ss_transforms as sstr
from config.enums import DatasetOptions
from datasets.idda import IDDADataset
from fed_setting.snapshot import Snapshot
from loggers.logger import BaseDecorator
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics

# Abstarct Factory

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

class DatasetFactory(ABC):

    def __init__(self, root:str, train_transforms: sstr.Compose, test_transforms: sstr.Compose) -> None:
        self.root = root
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    @abstractmethod
    def construct(self) -> List[BaseDataset]:
        pass

    @abstractmethod
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        pass

    @abstractmethod
    def construct_test_dataset(self) -> List[BaseDataset]:
        pass

class Experiment(ABC):

    @abstractmethod
    def train(self, starting:int = 0) -> int:
        pass

    @abstractmethod
    def eval_train(self) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass

    @abstractmethod
    def save(self) -> Snapshot:
        pass

    @abstractmethod
    def load_snapshot(self, snapshot: Snapshot) -> int:
        pass

class ExperimentFactory(ABC):

    def __init__(self,
                 args: Namespace, 
                 train_datasets: List[VisionDataset], 
                 test_datasets: List[VisionDataset], 
                 model: _SimpleSegmentationModel, 
                 metrics: Dict[str, StreamSegMetrics],
                 reduction: Callable[[Any], Any],
                 optimizer_factory: OptimizerFactory,
                 scheduler_factory: SchedulerFactory,
                 logger: BaseDecorator) -> None:
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.model = model
        self.metrics = metrics
        self.reduction = reduction
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.n_epochs = args.num_epochs
        self.batch_size = args.bs
        self.logger = logger

    @abstractmethod
    def construct(self) -> Experiment:
        pass