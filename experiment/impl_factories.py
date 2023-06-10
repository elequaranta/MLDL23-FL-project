from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from numpy.typing import ArrayLike
from overrides import override
from sklearn.cluster import KMeans
import torch
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel
from torch.utils.data import DataLoader
from datasets.base_dataset import BaseDataset
from datasets.silo_idda import SiloIddaDataset

from experiment.centralized_model import CentralizedModel
from experiment.abs_factories import Experiment, ExperimentFactory
from experiment.client import Client
from experiment.server import Server
from experiment.silo_client import SiloClient
from experiment.silo_server import SiloServer
from experiment.sl_client import ClientSelfLearning
from experiment.sl_server import ServerSelfLearning
from federated.fed_params import FederatedServerParamenters
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory, SchedulerFactory
from utils.stream_metrics import StreamSegMetrics
from utils.utils import DistillationLoss

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
        train_clients, test_clients = self._gen_clients()
        self.fed_params = FederatedServerParamenters(args.num_rounds, 
                                               args.clients_per_round, 
                                               train_clients, 
                                               test_clients, 
                                               self.model)

    def construct(self) -> Experiment:
        server = Server(n_rounds=self.fed_params.n_rounds,
                        n_clients_round=self.fed_params.n_clients_round,
                        train_clients=self.fed_params.training_clients,
                        test_clients=self.fed_params.test_clients,
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
    
class FederatedSelfLearningFactory(FederatedFactory):

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
                         logger)
        self.n_rounds=args.num_rounds
        self.n_clients_round=args.clients_per_round
        self.n_round_teacher_model=args.update_teach
        self.confidence_threshold=args.conf_threshold
    
    @override
    def construct(self) -> Experiment:
        server = ServerSelfLearning(fed_params=self.fed_params,
                                    metrics=self.metrics, 
                                    optimizer_factory=self.optimizer_factory,
                                    logger=self.logger,
                                    n_round_teacher_model=self.n_round_teacher_model, 
                                    confidence_threshold=self.confidence_threshold)
        return server
    
    @override
    def _gen_clients(self) -> Tuple[List[ClientSelfLearning], List[Client]]:
        clients = [[], []]
        for ds in self.train_datasets:
            clients[0].append(ClientSelfLearning(n_epochs=self.n_epochs, 
                                                 batch_size=self.batch_size,
                                                 reduction=self.reduction, 
                                                 dataset=ds,
                                                 model=self.model,
                                                 optimizer_factory=self.optimizer_factory,
                                                 scheduler_factory=self.scheduler_factory))
        for ds in self.test_datasets:
            clients[1].append(Client(n_epochs=self.n_epochs, 
                                     batch_size=self.batch_size,
                                     reduction=self.reduction, 
                                     dataset=ds,
                                     model=self.model,
                                     optimizer_factory=self.optimizer_factory,
                                     scheduler_factory=self.scheduler_factory,
                                     test_client=True))
        return clients[0], clients[1]
    
class SiloLearningFactory(FederatedSelfLearningFactory):

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
                         logger)
        self.criterion = DistillationLoss(model=model,
                                          alpha=args.alpha,
                                          beta=args.beta,
                                          tau=args.tau)

    @override
    def construct(self) -> Experiment:
        server = SiloServer(fed_params=self.fed_params,
                            metrics=self.metrics, 
                            optimizer_factory=self.optimizer_factory,
                            logger=self.logger,
                            n_round_teacher_model=self.n_round_teacher_model, 
                            confidence_threshold=self.confidence_threshold)
        return server

    #TODO: test
    @override
    def _gen_clients(self) -> Tuple[List[SiloClient], List[Client]]:
        clients = [[], []]
        n_cluster = 5
        kmeans = KMeans(n_cluster)
        clusters_idx = self._get_cluster_id(self.train_datasets, kmeans, is_fit=True)
        for ds, clusters_id in zip(self.train_datasets, clusters_idx):
            clients[0].append(SiloClient(n_epochs=self.n_epochs, 
                                         batch_size=self.batch_size,
                                         reduction=self.reduction, 
                                         dataset=ds,
                                         model=self.model,
                                         optimizer_factory=self.optimizer_factory,
                                         scheduler_factory=self.scheduler_factory,
                                         cluster_id=clusters_id,
                                         criterion=self.criterion))
        clusters_idx = self._get_cluster_id(self.test_datasets, kmeans, is_fit=False)
        for ds, cluster_id in zip(self.test_datasets, clusters_idx):
            clients[1].append(SiloClient(n_epochs=self.n_epochs, 
                                     batch_size=self.batch_size,
                                     reduction=self.reduction, 
                                     dataset=ds,
                                     model=self.model,
                                     optimizer_factory=self.optimizer_factory,
                                     scheduler_factory=self.scheduler_factory,
                                     cluster_id=cluster_id,
                                     test_client=True,
                                     criterion=self.criterion))
        return clients[0], clients[1]
    
    def _get_cluster_id(self, dss: List[SiloIddaDataset], kmeans: KMeans, is_fit: bool) -> ArrayLike:
        styles = [ds.style.flatten() for ds in dss]
        features = np.vstack(styles)
        if is_fit:
            return kmeans.fit_predict(features)
        return kmeans.predict(features)
    
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