from argparse import Namespace
import copy
from typing import OrderedDict, Tuple
import torch
import wandb

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torchvision.datasets import VisionDataset
from utils.stream_metrics import StreamSegMetrics

from utils.utils import HardNegativeMining, MeanReduction, get_scheduler


class Client:

    def __init__(self,
                 args: Namespace, 
                 dataset: VisionDataset, 
                 model: DeepLabV3, 
                 test_client=False):
        
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError
    
    def _configure_optimizer(self):
        params = [{"params": filter(lambda p: p.requires_grad, self.model.parameters()),
                        'weight_decay': self.args.weight_decay}]
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay, nesterov=True)
        elif self.args.optimizer == "Adam":
            optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError("Select a type of optimizer already implemented")
        
        scheduler = get_scheduler(self.args, optimizer,
                                  max_iter=10 * self.args.num_epochs * len(self.train_loader))
        return optimizer, scheduler

    def run_epoch(self, 
                  cur_epoch: int, 
                  optimizer: optim.Optimizer, 
                  scheduler: _LRScheduler = None) -> torch.Tensor:
        """This method locally trains the model with the dataset of the client. 
            It handles the training at mini-batch level.
        Args:
            cur_epoch (int): current epoch of training
            optimizer (optim.Optimizer): optimizer used for the local training
            scheduler (_LRScheduler, optional): scheduler used for the local training. Defaults to None.
        Returns:
            Tensor: loss of the epoch
        """
        example_ct = 0
        for cur_step, (images, labels) in enumerate(self.train_loader):

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

            # Log loss every 5 batch
            #if ((cur_step + 1) % 5 == 0):
            #    wandb.log({"epoch": cur_epoch, f"{self.name}-loss": loss}, step=example_ct)
        return loss

    def train(self) -> Tuple[int, OrderedDict, torch.Tensor]:
        """This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        
        Returns:
            Tuple[int, OrderedDict, Tensor]: length of the local dataset, copy of the model parameters and loss of the training
        """
        num_train_samples = len(self.dataset)
        optimizer, scheduler = self._configure_optimizer()
        self.model.to(self.device)
        self.model.train()
        #wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        for epoch in range(self.args.num_epochs):
            loss = self.run_epoch(epoch, optimizer=optimizer, scheduler=scheduler)
        
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
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self.model(images)
                self.update_metric(metric, outputs["out"], labels)

    def update_model(self, params_dict: OrderedDict) -> None:
        self.model.load_state_dict(params_dict)
