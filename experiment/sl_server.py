
from copy import copy
import enum
import itertools
import math
from typing import Dict, List
from overrides import override
from torch import Tensor
from torch.nn import Threshold
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from tqdm import tqdm
from datasets.base_dataset import BaseDataset

from experiment.client import Client
from experiment.server import Server
from experiment.sl_client import ClientSelfLearning
from experiment.snapshot import Snapshot, SnapshotImpl
from loggers.logger import BaseDecorator
from models.abs_factories import OptimizerFactory
from utils.stream_metrics import StreamSegMetrics

class ServerSelfLearning(Server):

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
                 confidence_threshold: float
                 ) -> None:
        super().__init__(n_rounds, 
                         n_clients_round,
                         train_clients, 
                         test_clients, 
                         model,
                         optimizer_factory,
                         metrics,
                         logger)
        self.teacher_model = copy.deepcopy(model)
        self.n_round_teacher_model = n_round_teacher_model
        self.threshold = Threshold(confidence_threshold, value=-1)
        self.update_client_ds(n_round_teacher_model, train_clients)

    @override
    def train(self, starting:int = 0) -> int:
        self.rounds_trained = starting
        self.logger.watch(self.model, log="all", log_freq=10)
        n_exaple = 0
        n_rounds_between_snap = math.ceil(self.n_rounds - starting / self.N_CHECKPOINTS_RUN)
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
            for client in clients:
                labels = [
                    self.get_label_from_pred(self.teacher_model(img)["out"]) 
                    for img, _ in client.dataset]
                client.dataset.update_labels(labels)
                # for img, _ in client.dataset:
                #     out = self.teacher_model(img)
                #     lbl = self.get_label_from_pred(out["out"])
                #     labels.append(lbl)
                    

    def get_label_from_pred(self, prediction) -> Tensor:
        class_pred = prediction.max(dim=0)
        return self.threshold(class_pred)

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
        self.model.load_state_dict(
                    state.get(ServerSelfLearning.ServerSelfLearningStateKey.MODEL_DICT))
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.teacher_model.load_state_dict( \
                    state.get(ServerSelfLearning.ServerSelfLearningStateKey.TEACHER_DICT, \
                    copy.deepcopy(self.model_params_dict)))
        self.optimizer.load_state_dict(state.get(ServerSelfLearning.ServerSelfLearningStateKey.OPTIMIZER_DICT))
        self.n_clients_round = state.get(ServerSelfLearning.ServerSelfLearningStateKey.CLIENTS_ROUND, self.n_clients_round)
        return state.get(ServerSelfLearning.ServerSelfLearningStateKey.ROUND, 0)
    
    class ServerSelfLearningStateKey(enum.Enum):
        MODEL_DICT = "model_dict"
        TEACHER_DICT = "teacher_dict"
        OPTIMIZER_DICT = "optimizer_dict"
        ROUND = "round_number"
        CLIENTS_ROUND = "clients_round"
