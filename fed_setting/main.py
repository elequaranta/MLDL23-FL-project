from argparse import Namespace
from typing import Any, Dict, List
import wandb
from torchvision.datasets import VisionDataset

from centr_setting.centralized_model import CentralizedModel
from utils.init_fs import Serializer
from utils.stream_metrics import StreamSegMetrics
from client import Client
from server import Server


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]

def main(
        args: Namespace, 
        train_datasets: List[VisionDataset], 
        test_datasets: Dict[str, VisionDataset], 
        model: Any, 
        metrics: Dict[str, StreamSegMetrics], 
        serializer: Serializer):

    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()

if __name__ == '__main__':
    main()