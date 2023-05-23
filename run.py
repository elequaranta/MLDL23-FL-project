# Imports from libraries
import math
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
from datasets.utils import get_dataset_num_classes
from models.deeplabv3 import deeplabv3_mobilenetv2

# Imports from our code base

from factories.abstract_factories import *
from factories.impl_factories import *
from config.enums import ExperimentPhase, ModelOptions, OptimizerOptions, SchedulerOptions
from config.args import get_parser
from utils.stream_metrics import StreamClsMetrics, StreamSegMetrics
from utils.utils import HardNegativeMining, MeanReduction
import datasets.ss_transforms as sstr
from loggers.logger import DummyLogger, LocalLoggerDecorator, Logger, WandbLoggerDecorator

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def get_args() -> Namespace:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    return args

def get_transforms(args: Namespace) -> Tuple[sstr.Compose, sstr.Compose]:
    print('Generating Transformation... \U0001F975')
    return TransformsFactory(args).construct()

def get_datasets(args: Namespace, train_transforms: sstr.Compose, test_transforms: sstr.Compose) \
    -> Tuple[List[VisionDataset], List[VisionDataset]]:
    print('Generating Datasets... \U0001F975')
    match args.dataset:
        case DatasetOptions.IDDA:
            return IddaDatasetFactory(args.framework, train_transforms, test_transforms).construct()
        case _:
            raise NotImplementedError("The dataset chosen is not implemented")
        
def get_model(args: Namespace):
    print(f'Initializing Model... \U0001F975')
    match args.model:
        case ModelOptions.DEEPLABv3_MOBILENETv2:
            return DeepLabV3MobileNetV2Factory(dataset_type=args.dataset).construct()
        case _:
            raise NotImplementedError("The model chosen is not implemented")
        
def get_optimizer(args: Namespace, model) -> Tuple[Optimizer, OptimizerFactory]:
    print('Initializing Optimizer... \U0001F975')
    match args.optimizer:
        case OptimizerOptions.SGD:
            optimizer_factory = SGDFactory(args.lr, args.weight_decay, args.momentum, model.parameters())
            optimizer = optimizer_factory.construct()
        case OptimizerOptions.ADAM:
            optimizer_factory = AdamFactory(args.lr, args.weight_decay, model.parameters())
            optimizer = optimizer_factory.construct()
        case _:
            raise NotImplementedError("The optimizer chosen is not implemented")
    return optimizer, optimizer_factory

def get_scheduler_factory(args: Namespace, len_train_dataset: int, optimizer: Optimizer) \
    -> SchedulerFactory:
    print('Generating Scheduler... \U0001F975')
    match args.lr_policy:
        case SchedulerOptions.POLY:
            max_iter = math.floor(args.num_epochs * (len_train_dataset / args.bs))
            return LambdaSchedulerFactory(args.lr_power, optimizer, max_iter)
        case SchedulerOptions.STEP:
            return StepLRSchedulerFactory(args.lr_decay_step, args.lr_decay_factor, optimizer)
        case _:
            raise NotImplementedError("The scheduler chosen is not implemented")
        
def get_logger(args: Namespace) -> Logger:
    logger = DummyLogger(args)
        
    if not args.not_use_wandb:
        logger = WandbLoggerDecorator(logger)
    if not args.not_use_local_logging:
        logger = LocalLoggerDecorator(logger)

    return logger

def init_env():
    try:

        start = time.time()
        args = get_args()
        logger = get_logger(args)
        train_transforms, test_transforms = get_transforms(args)
        train_datasets, test_datasets = get_datasets(args, train_transforms, test_transforms)
        model = get_model(args)
        # Generation Reduction
        if args.hnm:
            reduction = HardNegativeMining()
        else:
            reduction = MeanReduction()
        optimizer, optimizer_factory = get_optimizer(args, model)
        scheduler_factory = get_scheduler_factory(args, len(train_datasets[0]), optimizer)
        metrics = set_metrics(args)
        
        match args.framework:
            case 'federated':
                experiment = FederatedFactory(args=args, 
                                 train_datasets=train_datasets, 
                                 test_datasets=test_datasets, 
                                 model=model, 
                                 metrics=metrics, 
                                 reduction=reduction, 
                                 optimizer_factory=optimizer_factory, 
                                 scheduler_factory=scheduler_factory,
                                 logger=logger).construct()
            case 'centralized':
                experiment = CentralizedFactory(args=args, 
                                 train_datasets=train_datasets, 
                                 test_datasets=test_datasets, 
                                 model=model, 
                                 metrics=metrics, 
                                 reduction=reduction, 
                                 optimizer_factory=optimizer_factory, 
                                 scheduler_factory=scheduler_factory,
                                 logger=logger).construct()
            case _:
                raise NotImplementedError("The framework chosen is not implemented")

        match args.phase:
            case ExperimentPhase.ALL:
                experiment.train()
                snapshot = experiment.save()
                logger.save(snapshot)
                experiment.eval_train()
                experiment.test()
            case ExperimentPhase.TRAIN:
                experiment.train()
            case ExperimentPhase.TEST:
                experiment.eval_train()
                experiment.test()
            case _:
                raise NotImplementedError("The phase chosen is not implemented")

        end = time.time()
        print(f"Elapsed time: {round(end - start, 2)}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    init_env()