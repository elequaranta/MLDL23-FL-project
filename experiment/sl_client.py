from typing import Any, Callable
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDataset
from datasets.sl_idda import IDDADatasetSelfLearning
from experiment.client import Client
from models.abs_factories import OptimizerFactory, SchedulerFactory


class ClientSelfLearning(Client):

    def __init__(self, 
                 n_epochs: int, 
                 batch_size: int, 
                 reduction: Callable[[Any], Any], 
                 dataset: IDDADatasetSelfLearning, 
                 model: DeepLabV3, 
                 optimizer_factory: OptimizerFactory, 
                 scheduler_factory: SchedulerFactory, 
                 test_client=False):
        super().__init__(n_epochs, 
                         batch_size, 
                         reduction, 
                         dataset, 
                         model, 
                         optimizer_factory, 
                         scheduler_factory, 
                         test_client)
        
    def update_metric(self, metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        prediction = self.dataset.convert_class(prediction)
        metric.update(labels, prediction)
