from typing import Any, Callable, List, TypeVar
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from datasets.base_dataset import BaseDataset

from experiment.client import Client
from models.abs_factories import OptimizerFactory, SchedulerFactory

T = TypeVar("T", bound=Client)
U = TypeVar("U", bound=Client)
D = TypeVar("D", bound=BaseDataset)

class FederatedServerParamenters():

    def __init__(self, 
                 n_rounds: int,
                 n_clients_round: int,
                 training_clients: List[T],
                 test_clients: List[U],
                 model: DeepLabV3) -> None:
        self._n_rounds = n_rounds
        self._n_clients_round = n_clients_round
        self._training_clients = training_clients
        self._test_clients = test_clients
        self._model = model

    @property
    def n_rounds(self) -> int:
        return self._n_rounds
    
    @property
    def n_clients_round(self) -> int:
        return self._n_clients_round
    
    @property
    def training_clients(self) -> List[T]:
        return self._training_clients
    
    @property
    def test_clients(self) -> List[U]:
        return self._test_clients

    @property
    def model(self) -> DeepLabV3:
        return self._model
    
class FederatedClientParameters():

    def __init__(self, 
                 n_epochs: int, 
                 batch_size: int,
                 reduction: Callable[[Any], Any], 
                 dataset: D,
                 model: DeepLabV3,
                 optimizer_factory: OptimizerFactory,
                 scheduler_factory: SchedulerFactory) -> None:
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._reduction = reduction
        self._dataset = dataset
        self._model = model
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory

    @property
    def n_epochs(self) -> int:
        return self._n_epochs
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def reduction(self) -> int:
        return self._reduction
    
    @property
    def dataset(self) -> int:
        return self._dataset
    
    @property
    def model(self) -> int:
        return self._model
    
    @property
    def optimizer_factory(self) -> int:
        return self._optimizer_factory
    
    @property
    def scheduler_factory(self) -> int:
        return self._scheduler_factory