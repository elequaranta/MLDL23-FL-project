from argparse import Namespace
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

from torch import optim
from fed_setting.client import Client
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

from utils.stream_metrics import StreamSegMetrics


class Server:

    def __init__(self, 
                 args: Namespace, 
                 train_clients: List[Client], 
                 test_clients: List[Client], 
                 model: DeepLabV3, 
                 metrics: Dict[str, StreamSegMetrics]):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.optimizer = self._configure_optimizer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def select_clients(self) -> Client | np.ndarray[Client]:
        """Select a random subset of clients from the pool

        Returns:
            Client | np.ndarray[Client]: collection of clients extracted
        """
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients: np.ndarray[Client]) -> List[Tuple[int, OrderedDict]]:
        """This method trains the model with the dataset of the clients. It handles the training at single round level

        Args:
            clients (np.ndarray[Client]): list of all the clients to train

        Returns:
            List[Tuple[int, OrderedDict]]: number of samples used in the client & 
                                           state_dict of clients' model
        """
        updates = []
        self.optimizer.zero_grad()
        for i, c in enumerate(clients):
            self._load_server_model_on_client(c)
            num_samples, state_dict = c.train()
            
            update = self._compute_client_delta(state_dict)
            updates.append((num_samples, update))

        return updates

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

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        for r in range(self.args.num_rounds):
            clients = self.select_clients()
            updates = self.train_round(clients)
            self.update_model(updates)
            

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        metric = self.metrics["eval_train"]
        for client in self.train_clients:
            metric.reset()
            self._test(client, metric)

    def test(self):
        """
            This method handles the test on the test clients
        """
        for client in self.test_clients:
            self.metrics[client.name].reset()
            self._test(client, self.metrics[client.name])
    
    def _test(self, client: Client, metric: StreamSegMetrics) -> None:
        client.test(metric)
        print(metric)


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


    def _configure_optimizer(self):
        params = [{"params": filter(lambda p: p.requires_grad, self.model.parameters()),
                        'weight_decay': self.args.weight_decay}]
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay, nesterov=True)
        elif self.args.optimizer == "Adam":
            optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError("Select a type of optimizer already implemented")
        
        return optimizer