# Imports from libraries
import random
import time
import wandb
import torch
import importlib
import numpy as np
from argparse import Namespace
from typing import Any, Dict, List, Tuple
from torch import nn
from torchvision.datasets import VisionDataset
from torchvision.models import resnet18
from models.deeplabv3 import deeplabv3_mobilenetv2

# Imports from our code base
from datasets.utils import get_dataset_num_classes, get_datasets
from utils.args import get_parser
from utils.init_fs import Serializer
from utils.stream_metrics import StreamClsMetrics, StreamSegMetrics

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        # TODO: missing code here!
        raise NotImplementedError
    raise NotImplementedError

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics    

def init_env():
    try:
        parser = get_parser()
        args = parser.parse_args()
        set_seed(args.seed)

        if not args.not_use_wandb:
            wandb.init(
                project=args.project, 
                name=args.exp_name, 
                config=vars(args))

        start = time.time()

        print(f'Initializing model...')
        model = model_init(args)
        print('Done.')

        print('Generate datasets...')
        train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        metrics = set_metrics(args)
        serializer = Serializer(args.exp_name, args.not_use_serializer)
        serializer.save_params(vars(args))

        if args.framework == 'federated':
            main_module = 'fed_setting.main'
            main = getattr(importlib.import_module(main_module), 'main')
            main(args, train_datasets, test_datasets.values(), model, metrics, serializer)
        elif args.framework == 'centralized':
            main_module = 'centr_setting.main'
            main = getattr(importlib.import_module(main_module), 'main')
            main(args, train_datasets, test_datasets, model, metrics, serializer)
        else:
            raise NotImplementedError

        end = time.time()
        print(f"Elapsed time: {round(end - start, 2)}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    init_env()