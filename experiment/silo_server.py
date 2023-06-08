

from collections import OrderedDict
import copy
import enum
import itertools
from typing import Dict, List, Tuple
from overrides import override
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from experiment.client import Client
from experiment.sl_client import ClientSelfLearning
from experiment.sl_server import ServerSelfLearning
from experiment.snapshot import Snapshot, SnapshotImpl
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory
from utils.stream_metrics import StreamSegMetrics


class SiloServer(ServerSelfLearning):

    def __init__(self, 
                 n_rounds: int, 
                 n_clients_round: int, 
                 train_clients: List[ClientSelfLearning], 
                 test_clients: List[Client], 
                 model: DeepLabV3, 
                 optimizer_factory: OptimizerFactory, 
                 metrics: Dict[str, StreamSegMetrics], 
                 logger: BaseDecorator, 
                 n_round_teacher_model: int, 
                 confidence_threshold: float) -> None:
        super().__init__(n_rounds, 
                         n_clients_round, 
                         train_clients, 
                         test_clients, 
                         model, 
                         optimizer_factory, 
                         metrics, 
                         logger, 
                         n_round_teacher_model, 
                         confidence_threshold)
        
    # @override
    # def aggregate(self, updates: List[Tuple[int, OrderedDict]]) -> OrderedDict:

    #     total_weight = 0.
    #     base = OrderedDict()

    #     for (client_samples, client_model) in updates:

    #         total_weight += client_samples
    #         for key, value in client_model.items():
    #             if 'bn.running' not in key and 'bn.num_batches_tracked' not in key:
    #                 if key in base:
    #                     base[key] += client_samples * value.type(torch.FloatTensor)
    #                 else:
    #                     base[key] = client_samples * value.type(torch.FloatTensor)
    #     averaged_sol_n = copy.deepcopy(self.model_params_dict)
    #     for key, value in base.items():
    #         if total_weight != 0:
    #             averaged_sol_n[key] = value.to('cuda') / total_weight

    #     return averaged_sol_n

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
        self.model.load_state_dict(
                    state.get(SiloServer.SiloServerStateKey.MODEL_DICT))
        self.model.to(self.device)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.teacher_model.load_state_dict( \
                    state.get(SiloServer.SiloServerStateKey.TEACHER_DICT, \
                    copy.deepcopy(self.model_params_dict)))
        self.teacher_model.to(self.device)
        for client in itertools.chain(self.train_clients, self.test_clients):
            client.criterion.update_model(copy.deepcopy(self.model_params_dict))
        self.optimizer.load_state_dict(state.get(SiloServer.SiloServerStateKey.OPTIMIZER_DICT))
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