from argparse import Namespace
import copy
import enum
import math
from typing import Any, Callable, Dict, List
from overrides import override
import torch

from torch import Tensor, optim, nn
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from tqdm import tqdm
import wandb
from factories.abstract_factories import Experiment, OptimizerFactory, SchedulerFactory
from fed_setting.snapshot import Snapshot, SnapshotImpl
from loggers.logger import BaseDecorator
from utils.stream_metrics import StreamSegMetrics


class CentralizedModel(Experiment):

    def __init__(self, 
                 n_epochs: int,
                 batch_size: int,
                 reduction: Callable[[Any], Any],
                 train_dataset: VisionDataset, 
                 test_datasets: Dict[str,VisionDataset], 
                 model: DeepLabV3,
                 metrics: Dict[str, StreamSegMetrics],
                 optimizer_factory: OptimizerFactory,
                 scheduler_factory: SchedulerFactory,
                 logger: BaseDecorator):
        
        self.n_epochs = n_epochs
        self.model = model
        self.train_dataset = train_dataset
        self.test_datasets = test_datasets
        self.logger = logger
        self.metrics = metrics

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for dataset in test_datasets:
            if dataset.client_name == "test_same_dom":
                self.test_same_dom_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            else:
                self.test_diff_dom_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = reduction
        self.optimizer = optimizer_factory.construct()
        self.scheduler = scheduler_factory.construct()
    
    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def run_epoch(self, cur_epoch: int, optimizer: optim.Optimizer, scheduler: _LRScheduler = None):
        """Train the model using mini-batch of training data

        Args:
            cur_epoch (int): current epoch
            optimizer (optim.Optimizer): optimizer used in the back propagation
            scheduler (_LRScheduler, optional): scheduler used in the back propagation. Defaults to None.
        """
        example_ct = 0
        batch_ct = 0
        for _, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.reduction(self.criterion(outputs['out'], labels), labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            example_ct += len(images)
            batch_ct += 1

            # Log loss every 5 batch
            if ((batch_ct) % 5 == 0):
                self.logger.log({"epoch": cur_epoch, "loss": loss, "step": example_ct})

    @override
    def train(self, starting:int = 0) -> int:
        """This method locally trains the model with the idda dataset. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        
        Returns:
            int: length of the dataset used in training
        """
        tot_num_samples = len(self.train_dataset)
        self.model.to(self.device)
        self.model.train()
        self.logger.watch(self.model, self.criterion, log="all", log_freq=10)
        for epoch in tqdm(range(starting, self.n_epochs)):
            self.run_epoch(cur_epoch=epoch+1, optimizer=self.optimizer, scheduler=self.scheduler)

        return tot_num_samples
    
    @override
    def eval_train(self):
        self.model.to(self.device)
        self.model.eval()
        metric = self.metrics["eval_train"]
        for i, (images, labels) in enumerate(self.test_train_loader):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            outputs = self.model(images)
            self.update_metric(metric, outputs["out"], labels)
        results = metric.get_results()
        self.logger.save_results(results)
        self.logger.summary({f"eval_train mIoU" : results["Mean IoU"]})

    @override
    def test(self):
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            for loader in [self.test_same_dom_loader, self.test_diff_dom_loader]:
                metric = self.metrics[loader.dataset.name]
                for i, (images, labels) in enumerate(loader):
                    images = images.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device, dtype=torch.long)
                    outputs = self.model(images)
                    self.update_metric(metric, outputs["out"], labels)
                results = metric.get_results()
                self.logger.save_results(results)
                self.logger.summary({f"{loader.dataset.name} mIoU" : results["Mean IoU"]})

    @override   
    def save(self) -> Snapshot:
        state = {
            CentralizedModel.CentralizedModelKey.MODEL_DICT: self.model.state_dict(),
            CentralizedModel.CentralizedModelKey.OPTIMIZER_DICT: self.optimizer.state_dict(),
            CentralizedModel.CentralizedModelKey.EPOCHS: self.n_epochs,
            CentralizedModel.CentralizedModelKey.SCHEDULER_DICT: self.scheduler.state_dict()
        }
        snapshot = SnapshotImpl(state, name="centr-trained")
        return snapshot

    @override
    def load_snapshot(self, snapshot: Snapshot) -> int:
        state = snapshot.get_state()
        self.model.load_state_dict(state[CentralizedModel.CentralizedModelKey.MODEL_DICT])
        self.optimizer.load_state_dict(state[CentralizedModel.CentralizedModelKey.OPTIMIZER_DICT])
        self.scheduler.load_state_dict(state[CentralizedModel.CentralizedModelKey.SCHEDULER_DICT])
        return state[CentralizedModel.CentralizedModelKey.EPOCHS]

    class CentralizedModelKey(enum.Enum):
        MODEL_DICT = "model_dict"
        OPTIMIZER_DICT = "optimizer_dict"
        EPOCHS = "n_epochs"
        SCHEDULER_DICT = "scheduler_dict"
