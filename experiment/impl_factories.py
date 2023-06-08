from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple
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
    
class FederatedSelfLearningFactory(ExperimentFactory):

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
        train_clients, test_clients = self._gen_clients()
        server = ServerSelfLearning(n_rounds=self.n_rounds,
                                    n_clients_round=self.n_clients_round,
                                    train_clients=train_clients,
                                    test_clients=test_clients,
                                    model=self.model,
                                    metrics=self.metrics, 
                                    optimizer_factory=self.optimizer_factory,
                                    logger=self.logger,
                                    n_round_teacher_model=self.n_round_teacher_model, 
                                    confidence_threshold=self.confidence_threshold)
        return server
    
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
    
class SiloLearningFactory(ExperimentFactory):

    def __init__(self, 
                 args: Namespace, 
                 train_datasets: List[SiloIddaDataset], 
                 test_datasets: List[BaseDataset], 
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
        train_clients, test_clients = self._gen_clients()
        server = SiloServer(n_rounds=self.n_rounds,
                            n_clients_round=self.n_clients_round,
                            train_clients=train_clients,
                            test_clients=test_clients,
                            model=self.model,
                            metrics=self.metrics, 
                            optimizer_factory=self.optimizer_factory,
                            logger=self.logger,
                            n_round_teacher_model=self.n_round_teacher_model, 
                            confidence_threshold=self.confidence_threshold)
        return server

    def _gen_clients(self) -> Tuple[List[SiloClient], List[Client]]:
        clients = [[], []]
        self._update_ds_mean_std_clusters(self.train_datasets)
        for ds in self.train_datasets:
            clients[0].append(SiloClient(n_epochs=self.n_epochs, 
                                         batch_size=self.batch_size,
                                         reduction=self.reduction, 
                                         dataset=ds,
                                         model=self.model,
                                         optimizer_factory=self.optimizer_factory,
                                         scheduler_factory=self.scheduler_factory,
                                         criterion=self.criterion))
        for ds in self.test_datasets:
            clients[1].append(SiloClient(n_epochs=self.n_epochs, 
                                     batch_size=self.batch_size,
                                     reduction=self.reduction, 
                                     dataset=ds,
                                     model=self.model,
                                     optimizer_factory=self.optimizer_factory,
                                     scheduler_factory=self.scheduler_factory,
                                     test_client=True,
                                     criterion=self.criterion))
        return clients[0], clients[1]
    
    def _update_ds_mean_std_clusters(dss: List[SiloIddaDataset]):
        n_cluster = 5
        m = torch.zeros(size=(len(dss), 3))
        s = torch.zeros(size=(len(dss), 3))
        for i, ds in enumerate(dss):
            m[i,:] = ds.mean
            s[i,:] = ds.std
        cluster_mean = torch.zeros(size=(n_cluster, 3))
        cluster_std = torch.zeros(size=(n_cluster, 3))
        kmeans = KMeans(n_cluster)
        features = torch.hstack((m, s))
        clusters_idx = kmeans.fit_predict(features)
        for i, v in enumerate(clusters_idx):
            cluster_mean[v, :] += m[i, :]
            cluster_std[v, :] += m[i, :]
        cluster_mean.div(m.size(0))
        cluster_std.div(s.size(0))
        for i, v in enumerate(clusters_idx):
            dss[i].mean = cluster_mean[v]
            dss[i].std = cluster_std[v]
        
    
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