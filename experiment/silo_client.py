

import copy
from typing import Any, Callable, OrderedDict
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from datasets.silo_idda import SiloIddaDataset
from experiment.sl_client import ClientSelfLearning
from models.abs_factories import OptimizerFactory, SchedulerFactory


class SiloClient(ClientSelfLearning):

    def __init__(self, 
                 n_epochs: int, 
                 batch_size: int, 
                 reduction: Callable[[Any], Any], 
                 dataset: SiloIddaDataset, 
                 model: DeepLabV3, 
                 optimizer_factory: OptimizerFactory, 
                 scheduler_factory: SchedulerFactory,
                 criterion: Callable[[Any], Any],
                 test_client=False):
        super().__init__(n_epochs, 
                         batch_size, 
                         reduction, 
                         dataset, 
                         model, 
                         optimizer_factory, 
                         scheduler_factory, 
                         test_client)
        self.criterion = criterion

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
            loss = self.reduction(self.criterion(outputs['out'], labels, images), labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss

    def update_model(self, params_dict: OrderedDict) -> None:
        params_dict["bn.running_mean"] = copy.deepcopy(self.dataset.mean)
        params_dict["bn.running_var"] = copy.deepcopy(self.dataset.std)
        self.model.load_state_dict(params_dict)
        self.optimizer_factory.update_model_params(self.model.parameters())
        self.optimizer = self.optimizer_factory.construct()
        self.scheduler_factory.update_optimizer(self.optimizer)
        self.scheduler = self.scheduler_factory.construct()