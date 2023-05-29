import copy
import torch
import numpy as np
from typing import Any, Callable, OrderedDict, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from datasets.base_dataset import BaseDataset

from models.abs_factories import OptimizerFactory, SchedulerFactory
from utils.stream_metrics import StreamSegMetrics

class Client:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 reduction: Callable[[Any], Any],
                 dataset: BaseDataset, 
                 model: DeepLabV3,
                 optimizer_factory: OptimizerFactory,
                 scheduler_factory: SchedulerFactory,
                 test_client=False):
        self.n_epochs = n_epochs
        self.dataset = dataset
        self._name = self.dataset.name
        self.model = model
        if not test_client:
            self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            self.data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.reduction = reduction
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.optimizer = self.optimizer_factory.construct()
        self.scheduler = self.scheduler_factory.construct()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return self.name
    
    @property
    def name(self) -> str:
        return self._name

    # @staticmethod
    def update_metric(self, metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu()
        prediction = np.array(self.dataset.convert_class()(prediction))
        metric.update(labels, prediction)

    def run_epoch(self, cur_epoch: int) -> torch.Tensor:
        """This method locally trains the model with the dataset of the client. 
            It handles the training at mini-batch level.
        Args:
            cur_epoch (int): current epoch of training
        Returns:
            Tensor: loss of the epoch
        """
        for cur_step, (images, labels) in enumerate(self.data_loader):

            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.reduction(self.criterion(outputs['out'], labels), labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss

    def train(self) -> Tuple[int, OrderedDict, torch.Tensor]:
        """This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        
        Returns:
            Tuple[int, OrderedDict, Tensor]: length of the local dataset, copy of the model parameters and loss of the training
        """
        num_train_samples = len(self.dataset)
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(self.n_epochs):
            loss = self.run_epoch(epoch)
        
        state_dict = copy.deepcopy(self.model.state_dict())
        return num_train_samples, state_dict, loss

    def test(self, metric: StreamSegMetrics):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader):
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self.model(images)
                self.update_metric(metric, outputs["out"], labels)

    def update_model(self, params_dict: OrderedDict) -> None:
        self.model.load_state_dict(params_dict)
        self.optimizer_factory.update_model_params(self.model.parameters())
        self.optimizer = self.optimizer_factory.construct()
        self.scheduler_factory.update_optimizer(self.optimizer)
        self.scheduler = self.scheduler_factory.construct()
