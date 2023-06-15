
import copy
import enum
import itertools
import math
from typing import Dict, List
from overrides import override
from torch import Tensor
import torch
from torch.nn import Threshold
from torch.nn.functional import softmax
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from tqdm import tqdm
from torch.utils.data import DataLoader

from experiment.client import Client
from experiment.server import Server
from experiment.sl_client import ClientSelfLearning
from experiment.snapshot import Snapshot, SnapshotImpl
from federated.fed_params import FederatedServerParamenters
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory
from utils.stream_metrics import StreamSegMetrics

class ServerSelfLearning(Server):

    def __init__(self,
                 fed_params: FederatedServerParamenters,
                 optimizer_factory: OptimizerFactory,
                 metrics: Dict[str, StreamSegMetrics],
                 logger: BaseDecorator,
                 n_round_teacher_model: int,
                 confidence_threshold: float
                 ) -> None:
        super().__init__(fed_params.n_rounds, 
                         fed_params.n_clients_round,
                         fed_params.training_clients, 
                         fed_params.test_clients, 
                         fed_params.model,
                         optimizer_factory,
                         metrics,
                         logger)
        self.teacher_model = copy.deepcopy(self.model)
        self.n_round_teacher_model = n_round_teacher_model
        self.threshold = Threshold(confidence_threshold, value=-1)
        self.update_client_ds(n_round_teacher_model, self.train_clients)

    @override
    def train(self, starting:int = 0) -> int:
        self.rounds_trained = starting
        n_exaple = 0
        n_rounds_between_snap = math.ceil(self.n_rounds - starting / self.N_CHECKPOINTS_RUN)
        self.logger.watch(self.model, log="all", log_freq=10)
        for r in tqdm(range(starting, self.n_rounds)):
            clients = self.select_clients()
            self.update_client_ds(r, clients)
            updates, losses = self.train_round(clients)
            self.update_model(updates)

            # Online many people weights the losses value with the size of the client's datatet
            # could it be usefull?
            #if ((r + 1) % 5 == 0):
            for k, v in losses.items():
                self.logger.log({f"{k}-loss": v["loss"], "step": r+1})

            if r+1 % n_rounds_between_snap == 0:
                self.rounds_trained = r + 1
                self.logger.save(self.save())

        for client in itertools.chain(self.train_clients, self.test_clients):
            self._load_server_model_on_client(client)
        
        for client in self.train_clients:
            n_exaple += len(client.dataset)

        return n_exaple
    
    def update_client_ds(self, round: int, clients: List[ClientSelfLearning]) -> None:
        if self.n_round_teacher_model == -1:
            return
        if round % self.n_round_teacher_model == 0:
            self.teacher_model.load_state_dict(self.model_params_dict)
            step = 0
            for client in clients:
                labels = []
                dl = DataLoader(client.dataset, batch_size=5, shuffle=False)
                for img, _ in dl:
                    img = img.to(self.device)
                    out = self.teacher_model(img)
                    step += img.size()[0]
                    lbl = self.get_label_from_pred(out["out"], step)
                    labels.extend(lbl)
                client.dataset.update_labels(labels)

    def get_label_from_pred(self, prediction: Tensor, step) -> Tensor:
        prediction = softmax(prediction, dim=1)
        values, idx_class_pred = prediction.max(dim=1)
        values_copy = values
        values = self.threshold(values)
        conf_max = values_copy.amax(dim=(1,2))
        conf_min = values_copy.amin(dim=(1,2))
        conf_mean = values_copy.sum(dim=(1,2)).div(values_copy.size(dim=1) * values_copy.size(dim=2))
        for ma, mi, me in zip(conf_max, conf_min, conf_mean):
            self.logger.log(data={"max-conf":ma, "min-conf":mi, "mean-conf":me, "step": step})
        idx_class_pred[values == -1] = -1
        return [idx_class_pred[i, :, :] for i in range(idx_class_pred.size(dim=0))]
    
    @override
    def test(self):
        """
            This method handles the test on the test clients
        """
        for client in self.test_clients:
            metric = self.client_metrics[client.name]
            metric.reset()
            self.aggregated_metrics[client.name]
            client.test(metric)
            self.aggregated_metrics[client.name].update(metric, client.name)
        test_metrics_keys = filter(lambda key: "test" in key or "target" in key, self.aggregated_metrics.keys())
        for metric_key in test_metrics_keys:
            test_metric = self.aggregated_metrics[metric_key]
            result = test_metric.calculate_results()
            self.logger.save_results(result)
            self.logger.summary({metric_key: result})
            print(test_metric)

    @override
    def save(self) -> Snapshot:
        server_state = {
            ServerSelfLearning.ServerSelfLearningStateKey.MODEL_DICT: self.model_params_dict,
            ServerSelfLearning.ServerSelfLearningStateKey.TEACHER_DICT: self.teacher_model.state_dict(),
            ServerSelfLearning.ServerSelfLearningStateKey.OPTIMIZER_DICT: self.optimizer.state_dict(),
            ServerSelfLearning.ServerSelfLearningStateKey.ROUND: self.n_rounds,
            ServerSelfLearning.ServerSelfLearningStateKey.CLIENTS_ROUND: self.n_clients_round
        }
        snapshot = SnapshotImpl(server_state, name=f"fed-server-self-learning-{self.rounds_trained}")
        return snapshot

    @override
    def load_snapshot(self, snapshot: Snapshot) -> int:
        state = snapshot.get_state()
        for k, v in state.items():
            if k.value == ServerSelfLearning.ServerSelfLearningStateKey.MODEL_DICT.value:
                self.model.load_state_dict(state.get(k))
        self.model.to(self.device)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.teacher_model.load_state_dict( \
                    state.get(ServerSelfLearning.ServerSelfLearningStateKey.TEACHER_DICT, \
                    copy.deepcopy(self.model_params_dict)))
        self.teacher_model.to(self.device)
        self.update_client_ds(self.n_round_teacher_model, self.train_clients)
        for k, v in state.items():
            if k.value == ServerSelfLearning.ServerSelfLearningStateKey.OPTIMIZER_DICT.value:
                self.optimizer.load_state_dict(state.get(k))
        self.optimizer_to(self.optimizer, self.device)
        self.n_clients_round = state.get(ServerSelfLearning.ServerSelfLearningStateKey.CLIENTS_ROUND, self.n_clients_round)
        return state.get(ServerSelfLearning.ServerSelfLearningStateKey.ROUND, 0)

    def optimizer_to(self, optim, device):
      for param in optim.state.values():
          # Not sure there are any global tensors in the state dict
          if isinstance(param, torch.Tensor):
              param.data = param.data.to(device)
              if param._grad is not None:
                  param._grad.data = param._grad.data.to(device)
          elif isinstance(param, dict):
              for subparam in param.values():
                  if isinstance(subparam, torch.Tensor):
                      subparam.data = subparam.data.to(device)
                      if subparam._grad is not None:
                          subparam._grad.data = subparam._grad.data.to(device)
    
    class ServerSelfLearningStateKey(enum.Enum):
        MODEL_DICT = "model_dict"
        TEACHER_DICT = "teacher_dict"
        OPTIMIZER_DICT = "optimizer_dict"
        ROUND = "round_number"
        CLIENTS_ROUND = "clients_round"
