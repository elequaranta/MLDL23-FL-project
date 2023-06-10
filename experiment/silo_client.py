

from collections import OrderedDict
import copy
from typing import Any, Callable, Tuple
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
                 cluster_id: int,
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
        self._cluster_id = cluster_id

        self.bn_stats_dict = OrderedDict()

        for k, v in self.model.state_dict().items():
            if "bn" in k:
                self.bn_stats_dict[k] = copy.deepcopy(v)

    @property
    def cluster_id(self):
        return self._cluster_id
    
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
        self.update_bn_mean_std()
        return num_train_samples, state_dict, loss
    
    def update_bn_mean_std(self):
        for k, v in self.model.state_dict().items():
            if "bn.running" in k or "bn.num_batches_tracked" in k:
                self.bn_stats_dict[k] = copy.deepcopy(v)

    def update_model(self, params_dict: OrderedDict, bn_weight_bias: OrderedDict) -> None:
        self.model.load_state_dict(params_dict)
        self.model.load_state_dict(bn_weight_bias, strict=False)
        self.model.load_state_dict(self.bn_stats_dict, strict=False)
        self.optimizer_factory.update_model_params(self.model.parameters())
        self.optimizer = self.optimizer_factory.construct()
        self.scheduler_factory.update_optimizer(self.optimizer)
        self.scheduler = self.scheduler_factory.construct()