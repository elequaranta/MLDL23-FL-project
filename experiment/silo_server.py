from collections import OrderedDict
import copy
import enum
import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from overrides import override
from torch.utils.data import DataLoader
from federated.fed_params import FederatedServerParamenters
from self_learning.self_learning_params import SelfLearningParams
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from experiment.client import Client
from experiment.silo_client import SiloClient
from experiment.sl_client import ClientSelfLearning
from experiment.sl_server import ServerSelfLearning
from experiment.snapshot import Snapshot, SnapshotImpl
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory
from utils.stream_metrics import StreamSegMetrics


class SiloServer(ServerSelfLearning):

    def __init__(self,
                 fed_params: FederatedServerParamenters, 
                 optimizer_factory: OptimizerFactory,
                 metrics: Dict[str, StreamSegMetrics], 
                 logger: BaseDecorator, 
                 self_learning_params: SelfLearningParams,
                 n_clusters: int) -> None:
        self.n_clusters = n_clusters
        self.bn_statics = OrderedDict()
        for i in range(n_clusters):
            self.bn_statics[i] = OrderedDict()
        for i in range(n_clusters):
            for k, v in fed_params.model.state_dict().items():
                if "bn" in k:
                    self.bn_statics[i][k] = copy.deepcopy(v)

        super().__init__(fed_params=fed_params,
                         optimizer_factory=optimizer_factory, 
                         metrics=metrics, 
                         logger=logger, 
                         n_round_teacher_model=self_learning_params.n_round_teacher_model, 
                         confidence_threshold=self_learning_params.confidence_threshold)

    # TODO: test
    @override
    def train_round(self, clients) -> Tuple[List[Tuple[int, OrderedDict]], Dict[str, Dict[torch.Tensor, int]]]:
        """This method trains the model with the dataset of the clients. It handles the training at single round level

        Args:
            clients (np.ndarray[Client]): list of all the clients to train

        Returns:
            Tuple[List[Tuple[int, OrderedDict]], Dict[str, Dict[torch.Tensor, int]]]: number of samples used in the client, 
                                           state_dict of clients' model and last loss of every client
        """
        updates = []
        self.optimizer.zero_grad()
        losses = {}
        for i, c in enumerate(clients):
            self._load_server_model_on_client(c)
            num_samples, state_dict, loss = c.train()
            losses[c.name] = {'loss': loss, 'num_samples': num_samples}
            update = self._compute_client_delta(state_dict)
            updates.append((num_samples, update, c.cluster_id))

        self.rounds_trained += 1

        return updates, losses
        

    # TODO: test is it's correct    
    @override
    def aggregate(self, updates: List[Tuple[int, OrderedDict, Optional[int]]]) -> OrderedDict:

        total_weight = 0.
        base = OrderedDict()
        bn_base = OrderedDict()
        for i in range(self.n_clusters):
            bn_base[i] = OrderedDict()
        cluster_weight = torch.zeros(self.n_clusters)
        for (client_samples, client_model, client_cluster) in updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if "bn" not in key:
                    if key in base:
                        base[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base[key] = client_samples * value.type(torch.FloatTensor)
                else:
                    cluster_weight[client_cluster] += client_samples
                    if key in bn_base:
                        bn_base[client_cluster][key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        bn_base[client_cluster][key] = client_samples * value.type(torch.FloatTensor)

        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to(self.device) / total_weight

        for cluster_id, bn_stat in bn_base.items():
            for key, value in bn_stat.items():
                #if len(value) != 0:
                self.bn_statics[cluster_id][key] = value.to(self.device) / cluster_weight[cluster_id]

        return averaged_sol_n
                


    # TODO: test
    def update_client_ds(self, round: int, clients: List[SiloClient]) -> None:
        if self.n_round_teacher_model == -1:
            return
        if round % self.n_round_teacher_model == 0:
            self.teacher_model.load_state_dict(self.model_params_dict)
            for i in range(self.n_clusters):
                step = 0
                for client in clients:
                    if client.cluster_id == i:
                        self.teacher_model.load_state_dict(self.bn_statics[i], strict=False)
                        labels = []
                        dl = DataLoader(client.dataset, batch_size=5, shuffle=False)
                        for img, _ in dl:
                            img = img.to(self.device)
                            out = self.teacher_model(img)
                            step += img.size()[0]
                            lbl = self.get_label_from_pred(out["out"], step)
                            labels.extend(lbl)
                        client.dataset.update_labels(labels)

    # TODO: test
    def _load_server_model_on_client(self, client: SiloClient) -> None:
        """Load the server model into the client

        Args:
            client (Client): client where load the model state dict
        """
        # use self.model_params_dict to pass the params (it's a deep copy)
        # to not create side effect during training
        weights_biases = OrderedDict()
        for k, v in self.bn_statics[client.cluster_id].items():
            if "weight" in k or "bias" in k:
                weights_biases[k] = copy.deepcopy(v)
        client.update_model(self.model_params_dict, self.bn_statics[client.cluster_id])

    @override
    def save(self) -> Snapshot:
        server_state = {
            SiloServer.SiloServerStateKey.MODEL_DICT: self.model_params_dict,
            SiloServer.SiloServerStateKey.TEACHER_DICT: self.teacher_model.state_dict(),
            SiloServer.SiloServerStateKey.DISTILLATION_DICT: self.train_clients[0].criterion.model.state_dict(),
            SiloServer.SiloServerStateKey.OPTIMIZER_DICT: self.optimizer.state_dict(),
            SiloServer.SiloServerStateKey.ROUND: self.n_rounds,
            SiloServer.SiloServerStateKey.CLIENTS_ROUND: self.n_clients_round
        }
        snapshot = SnapshotImpl(server_state, name=f"fed-server-silo-self-learning-{self.rounds_trained}")
        return snapshot

    @override
    def load_snapshot(self, snapshot: Snapshot) -> int:
        state = snapshot.get_state()
        for k, v in state.items():
            if k.value == SiloServer.SiloServerStateKey.MODEL_DICT.value:
                self.model.load_state_dict(state.get(k))
        self.model.to(self.device)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.teacher_model.load_state_dict( \
                    state.get(SiloServer.SiloServerStateKey.TEACHER_DICT, \
                    copy.deepcopy(self.model_params_dict)))
        self.teacher_model.to(self.device)
        self.update_client_ds(self.n_round_teacher_model, self.train_clients)
        for client in itertools.chain(self.train_clients, self.test_clients):
            client.criterion.update_model(copy.deepcopy(self.model_params_dict))
        for k, v in state.items():
            if k.value == SiloServer.SiloServerStateKey.OPTIMIZER_DICT.value:
                self.optimizer.load_state_dict(state.get(k))
        self.optimizer_to(self.optimizer, self.device)
        self.n_clients_round = state.get(SiloServer.SiloServerStateKey.CLIENTS_ROUND, self.n_clients_round)
        return state.get(SiloServer.SiloServerStateKey.ROUND, 0)
    
    class SiloServerStateKey(enum.Enum):
        MODEL_DICT = "model_dict"
        TEACHER_DICT = "teacher_dict"
        DISTILLATION_DICT = "distillation_dict"
        OPTIMIZER_DICT = "optimizer_dict"
        ROUND = "round_number"
        CLIENTS_ROUND = "clients_round"
