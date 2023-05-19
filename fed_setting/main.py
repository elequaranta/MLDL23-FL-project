from argparse import Namespace
from typing import Any, Dict, List, Tuple
import wandb
from torchvision.datasets import VisionDataset

from centr_setting.centralized_model import CentralizedModel
from fed_setting.client import Client
from fed_setting.server import Server
from utils.init_fs import Serializer
from utils.stream_metrics import StreamSegMetrics



def gen_clients(
        args: Namespace, 
        train_datasets: List[VisionDataset], 
        test_datasets:List[VisionDataset], model) \
            -> Tuple[List[Client], List[Client]]:
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]

def main(
        args: Namespace, 
        train_datasets: List[VisionDataset], 
        test_datasets: List[VisionDataset], 
        model: Any, 
        metrics: Dict[str, StreamSegMetrics], 
        serializer: Serializer):

    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()
    server.eval_train()
    server.test()

if __name__ == '__main__':
    main()