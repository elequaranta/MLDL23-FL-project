from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel
from torchvision.datasets import VisionDataset

from experiment.snapshot import Snapshot
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory, SchedulerFactory
from utils.stream_metrics import StreamSegMetrics

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