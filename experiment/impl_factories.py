from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple
from overrides import override
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel
from experiment.centralized_model import CentralizedModel

from experiment.abs_factories import Experiment, ExperimentFactory
from experiment.client import Client
from experiment.server import Server
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory, SchedulerFactory
from utils.stream_metrics import StreamSegMetrics

class FederatedFactory(ExperimentFactory):

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
        super().__init__(args,
                       train_datasets, 
                       test_datasets, 
                       model, 
                       metrics,
                       reduction,
                       optimizer_factory,
                       scheduler_factory,
                       logger=logger)
        self.n_rounds=args.num_rounds
        self.n_clients_round=args.clients_per_round

    def construct(self) -> Experiment:
        train_clients, test_clients = self._gen_clients()
        server = Server(n_rounds=self.n_rounds,
                        n_clients_round=self.n_clients_round,
                        train_clients=train_clients,
                        test_clients=test_clients,
                        model=self.model,
                        metrics=self.metrics, 
                        optimizer_factory=self.optimizer_factory,
                        logger=self.logger)
        return server

    def _gen_clients(self) -> Tuple[List[Client], List[Client]]:
        clients = [[], []]
        for i, datasets in enumerate([self.train_datasets, self.test_datasets]):
            for ds in datasets:
                clients[i].append(Client(n_epochs=self.n_epochs, 
                                        batch_size=self.batch_size,
                                        reduction=self.reduction, 
                                        dataset=ds,
                                        model=self.model,
                                        optimizer_factory=self.optimizer_factory,
                                        scheduler_factory=self.scheduler_factory,
                                        test_client=i == 1))
        return clients[0], clients[1]
    
class CentralizedFactory(ExperimentFactory):

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
        super().__init__(args,
                       train_datasets, 
                       test_datasets, 
                       model, 
                       metrics,
                       reduction,
                       optimizer_factory,
                       scheduler_factory,
                       logger=logger)

    def construct(self) -> Experiment:
        centr_model = CentralizedModel(n_epochs=self.n_epochs, 
                                       batch_size=self.batch_size,
                                       reduction=self.reduction,
                                       train_dataset=self.train_datasets[0], 
                                       test_datasets=self.test_datasets, 
                                       model=self.model,
                                       metrics=self.metrics,
                                       optimizer_factory=self.optimizer_factory,
                                       scheduler_factory=self.scheduler_factory,
                                       logger=self.logger)
        return centr_model