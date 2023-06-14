from typing import Any, Callable
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from datasets.silo_idda import SiloIddaDataset
from experiment.silo_client import SiloClient
from models.abs_factories import OptimizerFactory, SchedulerFactory
from utils.stream_metrics import StreamSegMetrics


class BasicSiloClient(SiloClient):

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
                         criterion, 
                         cluster_id, 
                         test_client)
        
    def test(self, metric: StreamSegMetrics):
        state = self.model.state_dict()
        for k, v in state.items():
            if "bn.running" in k or "bn.num_batches_tracked" in k:
                state[k] = None
            if "bn.track_running_stats" in k:
                state[k] = False
        super().test(metric)
