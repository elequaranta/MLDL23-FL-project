import copy
from collections import OrderedDict
import enum
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from overrides import override
import torch
import itertools

from tqdm import tqdm
from torch import optim
from factories.abstract_factories import Experiment, OptimizerFactory
from fed_setting.client import Client
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from fed_setting.snapshot import SnapshotImpl, Snapshot
from loggers.logger import BaseDecorator

from utils.stream_metrics import AggregatedFederatedMetrics, StreamSegMetrics


class Server(Experiment):

    def __init__(self, 
                 n_rounds: int,
                 n_clients_round: int,
                 train_clients: List[Client], 
                 test_clients: List[Client], 
                 model: DeepLabV3,
                 optimizer_factory: OptimizerFactory,
                 metrics: Dict[str, StreamSegMetrics],
                 logger: BaseDecorator):
        self.n_rounds = n_rounds
        self.n_clients_round = n_clients_round
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.client_metrics = metrics
        self.aggregated_metrics = {
            "eval_train": AggregatedFederatedMetrics("eval_train"),
            "test_same_dom": AggregatedFederatedMetrics("test_same_dom"),
            "test_diff_dom": AggregatedFederatedMetrics("test_diff_dom"),
        }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.optimizer = optimizer_factory.construct()
        self.logger = logger

    def select_clients(self):
        """Select a random subset of clients from the pool

        Returns:
            Client | np.ndarray[Client]: collection of clients extracted
        """
        num_clients = min(self.n_clients_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

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
            updates.append((num_samples, update))

        return updates, losses

    def aggregate(self, updates: List[Tuple[int, OrderedDict]]) -> OrderedDict:
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        total_weight = 0.
        base = OrderedDict()

        for (num_client_samples, client_model) in updates:

            total_weight += num_client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += num_client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = num_client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to(self.device) / total_weight
        return averaged_sol_n
    
    def update_model(self, updates: List[Tuple[int, OrderedDict]]) -> None:
        averaged_state_dicts = self.aggregate(updates)
        self._server_opt(averaged_state_dicts)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    @override
    def train(self, starting:int = 0) -> int:
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        n_exaple = 0
        self.logger.watch(self.model, log="all", log_freq=10)
        for r in tqdm(range(starting, self.n_rounds)):
            clients = self.select_clients()
            updates, losses = self.train_round(clients)    
            self.update_model(updates)

            # Online many people weights the losses value with the size of the client's datatet
            # could it be usefull?
            if ((r + 1) % 5 == 0):
                for k, v in losses.items():
                    self.logger.log({f"{k}-loss": v["loss"]}, step=r+1)

        for client in itertools.chain(self.train_clients, self.test_clients):
            self._load_server_model_on_client(client)
        
        for client in self.train_clients:
            n_exaple += len(client.dataset)

        #if torch.cuda.is_available():
        #    print(torch.cuda.memory_summary(device=None, abbreviated=False))
        return n_exaple

    @override
    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        metric = self.client_metrics["eval_train"]
        agr_metric = self.aggregated_metrics["eval_train"]
        for client in self.train_clients:
            metric.reset()
            client.test(metric)
            agr_metric.update(metric, client.name)
        self.logger.summary["eval_train"] = agr_metric.calculate_results()
        print(agr_metric)

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
        test_metrics_keys = filter(lambda key: "test" in key, self.aggregated_metrics.keys())
        for metric_key in test_metrics_keys:
            test_metric = self.aggregated_metrics[metric_key]
            self.logger.summary[metric_key] = test_metric.calculate_results()
            print(test_metric)

    @override
    def save(self) -> Snapshot:
        server_state = {
            Server.ServerStateKey.MODEL_DICT: self.model_params_dict,
            Server.ServerStateKey.OPTIMIZER_DICT: self.optimizer.state_dict(),
            Server.ServerStateKey.ROUND: self.n_rounds,
            Server.ServerStateKey.CLIENTS_ROUND: self.n_clients_round
        }
        snapshot = SnapshotImpl(server_state, name="fed-server")
        return snapshot

    @override
    def load_snapshot(self, snapshot: Snapshot) -> int:
        state = snapshot.get_state()
        self.model.load_state_dict(state[Server.ServerStateKey.MODEL_DICT])
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.optimizer.load_state_dict(state[Server.ServerStateKey.OPTIMIZER_DICT])
        self.n_clients_round = state[Server.ServerStateKey.CLIENTS_ROUND]
        return state[Server.ServerStateKey.ROUND]


    def _load_server_model_on_client(self, client: Client) -> None:
        """Load the server model into the client

        Args:
            client (Client): client where load the model state dict
        """
        # use self.model_params_dict to pass the params (it's a deep copy)
        # to not create side effect during training
        client.update_model(self.model_params_dict)

    def _server_opt(self, pseudo_gradient):

        # Pseudo gradient (taken from papaer: https://arxiv.org/pdf/2003.00295.pdf)
        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.optimizer.step()
        # Take state_dict value of running variables and buffers from the pseudo_gradient
        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _compute_client_delta(self, client_model: OrderedDict) -> OrderedDict:
        delta = OrderedDict.fromkeys(client_model.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), client_model.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta
    
    class ServerStateKey(enum.Enum):
        MODEL_DICT = "model_dict"
        OPTIMIZER_DICT = "optimizer_dict"
        ROUND = "round_number"
        CLIENTS_ROUND = "clients_round"
