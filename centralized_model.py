import copy
import math
from typing import Dict, List
import torch

from torch import Tensor, optim, nn
from torch.optim.lr_scheduler import _LRScheduler
from collections import defaultdict
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import wandb
from utils.init_fs import Serializer
from utils.stream_metrics import StreamSegMetrics

from utils.utils import HardNegativeMining, MeanReduction, get_scheduler


class CentralizedModel:

    def __init__(self, 
                 args, 
                 train_dataset: VisionDataset, 
                 test_datasets: Dict[str,VisionDataset], 
                 model: DeepLabV3,
                 serializer: Serializer):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.test_datasets = test_datasets
        self.serializer = serializer

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.bs, shuffle=True, drop_last=True)
        self.test_same_dom_loader = DataLoader(test_datasets["same_dom"], batch_size=1, shuffle=False)
        self.test_diff_dom_loader = DataLoader(test_datasets["diff_dom"], batch_size=1, shuffle=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        # TODO understand what is this....
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

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
        else:
            optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        scheduler = get_scheduler(self.args, optimizer,
                                  max_iter=10000 * self.args.num_epochs * len(self.train_loader))
        return optimizer, scheduler

    def run_epoch(self, cur_epoch: int, optimizer: optim.Optimizer, n_steps_per_epoch: int, scheduler: _LRScheduler = None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        example_ct = 0
        for cur_step, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs['out'], labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            example_ct += len(images)
            wandb_metrics = {"train/train_loss": loss, 
                       "train/epoch": (cur_step + 1 + (n_steps_per_epoch * cur_epoch)) / n_steps_per_epoch, 
                       "train/example_ct": example_ct}
            
        wandb.log(wandb_metrics)

    def train(self) -> int:
        """
        This method locally trains the model with the idda dataset. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the dataset used in training, copy of the model parameters
        """
        tot_num_samples = len(self.train_dataset)
        optimizer, scheduler = self._configure_optimizer()
        self.model.to(self.device)
        self.model.train()
        n_steps_per_epoch = math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size)
        
        for epoch in range(self.args.num_epochs):
            print(f"Current epoch: {epoch}")
            self.run_epoch(cur_epoch=epoch, optimizer=optimizer, n_steps_per_epoch=n_steps_per_epoch)

        self.serializer.save_model(model=self.model)
        return tot_num_samples

    def test(self, metric: StreamSegMetrics, dataset_type: str):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.to(self.device)
        self.model.eval()

        if dataset_type == "train":
            loader = self.train_loader
        elif dataset_type == "same_dom":
            loader = self.test_same_dom_loader
        elif dataset_type == "diff_dom":
            loader = self.test_diff_dom_loader
        else:
            raise NotImplementedError("Select the type of dataset to test the model")
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                outputs = self.model(images, test=True, use_test_resize=self.args.use_test_resize)
                self.update_metric(metric, outputs, labels)
            results = metric.get_results()
            results["dataset"] = dataset_type
            wandb.log({"validation": results})
            wandb.summary['mIoU'] = results["Mean IoU"]
            self.serializer.save_results(results)
