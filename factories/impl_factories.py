from abc import ABC, abstractmethod
from argparse import Namespace
import json
import os
from typing import Any, Callable, Dict, Tuple
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR
from torch.optim import SGD, Adam, Optimizer
from torchvision.datasets import VisionDataset
from torchvision.models.segmentation.deeplabv3 import _SimpleSegmentationModel
from centr_setting.centralized_model import CentralizedModel

from factories.abstract_factories import *
from config.enums import DatasetOptions, NormOptions
from factories.abstract_factories import Experiment
from fed_setting.client import Client
from fed_setting.server import Server
from loggers.logger import BaseDecorator
from models.deeplabv3 import deeplabv3_mobilenetv2
import datasets.ss_transforms as sstr
from utils.stream_metrics import StreamSegMetrics

# Implementation of the Factory

class SGDFactory(OptimizerFactory):

    def __init__(self, lr: float, weight_decay: float, momentum:float, model_params_iter) -> None:
        super().__init__(lr=lr, 
                         weight_decay=weight_decay, 
                         model_params=model_params_iter)
        self.momentum = momentum

    def construct(self) -> Optimizer:
        return SGD(self.params, 
                   lr=self.lr, 
                   momentum=self.momentum,
                   weight_decay=self.weight_decay,
                   nesterov=True)

class AdamFactory(OptimizerFactory):
    def __init__(self, lr: float, weight_decay: float, model_params_iter) -> None:
        super().__init__(lr=lr, 
                         weight_decay=weight_decay, 
                         model_params=model_params_iter)
        
    def construct(self) -> Optimizer:
        return Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
    
        
class LambdaSchedulerFactory(SchedulerFactory):

    def __init__(self, lr_power: float, optimizer, max_iter: int) -> None:
        super().__init__(optimizer)
        self.max_iter = max_iter
        self.lr_power = lr_power

    def construct(self) -> _LRScheduler:
        assert self.max_iter is not None, "max_iter necessary for poly LR scheduler"
        return LambdaLR(self.optimizer, lr_lambda=lambda cur_iter: (1 - cur_iter / self.max_iter) ** self.lr_power)
    
class StepLRSchedulerFactory(SchedulerFactory):

    def __init__(self, lr_decay_step: int, lr_decay_factor: float , optimizer) -> None:
        super().__init__(optimizer)
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor

    def construct(self) -> _LRScheduler:
        return StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_factor)
    
class DeepLabV3MobileNetV2Factory(ModelFactory):

    def __init__(self, dataset_type: DatasetOptions) -> None:
        super().__init__(dataset_type)
    
    def construct(self) -> _SimpleSegmentationModel:
        return deeplabv3_mobilenetv2(num_classes=self.dataset_class_number)
    
class IddaDatasetFactory(DatasetFactory):

    def __init__(self,
                 framework: str,
                 train_transforms,
                 test_transforms) -> None:
        super().__init__(train_transforms, test_transforms)
        self.root = "data/idda"
        self.framework = framework
        
    def construct(self) -> Tuple[List[VisionDataset], List[VisionDataset]]:
        train_datasets = []
        test_datasets = []
        
        match self.framework:
            case "centralized":
                with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                    all_data = f.readlines()
                    train_datasets.append(IDDADataset(root=self.root,
                                list_samples=all_data,
                                transform=self.train_transforms,
                                client_name="train"))
            case "federated":
                with open(os.path.join(self.root, 'train.json'), 'r') as f:
                    all_data = json.load(f)
                    for client_id in all_data.keys():
                        train_datasets.append(IDDADataset(root=self.root, 
                                                    list_samples=all_data[client_id], 
                                                    transform=self.train_transforms,
                                                    client_name=client_id))
        with open(os.path.join(self.root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_datasets.append(IDDADataset(root=self.root,
                                        list_samples=test_same_dom_data, 
                                        transform=self.test_transforms,
                                        client_name='test_same_dom'))
        with open(os.path.join(self.root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_datasets.append(IDDADataset(root=self.root,
                                        list_samples=test_diff_dom_data,
                                        transform=self.test_transforms,
                                        client_name='test_diff_dom'))
            
        return train_datasets, test_datasets
    
class TransformsFactory():

    def __init__(self, args: Namespace) -> None:
        self.rsrc_transform = args.rsrc_transform
        self.rrc_transform = args.rrc_transform
        self.jitter = args.jitter
        self.h_resize = args.h_resize
        self.w_resize = args.w_resize
        self.min_scale = args.min_scale
        self.max_scale = args.max_scale
        match args.norm:
            case NormOptions.EROS:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
            case NormOptions.CTS:
                self.mean = [0.5, 0.5, 0.5]
                self.std = [0.5, 0.5, 0.5]

    def construct(self) -> Tuple[sstr.Compose, sstr.Compose]:
        train_transform = []
        test_transform = []

        train_transform.append(sstr.RandomHorizontalFlip(0.5))
        
        if self.rsrc_transform:
            train_transform.append(
                sstr.RandomScaleRandomCrop(crop_size=(1024, 1856), scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
            train_transform.append(sstr.Resize(size=(self.h_resize, self.w_resize)))

        elif self.rrc_transform:
            train_transform.append(
                sstr.RandomResizedCrop((self.h_resize, self.w_resize), scale=(self.min_scale, self.max_scale)))
    
        if self.jitter:
            train_transform.append(sstr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
    
        train_transform = train_transform + [sstr.ToTensor(), sstr.Normalize(mean=self.mean, std=self.std)]
        train_transform = sstr.Compose(train_transform)

        test_transform = test_transform + [sstr.ToTensor(), sstr.Normalize(mean=self.mean, std=self.std)]
        test_transform = sstr.Compose(test_transform)

        return train_transform, test_transform
    
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
