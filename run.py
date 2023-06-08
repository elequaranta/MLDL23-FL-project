# Imports from libraries

import math
import random
import time
from typing import List, Tuple
import torch
from torch.optim import Optimizer
import numpy as np
from argparse import Namespace

# Imports from our code base

import datasets.base_dataset as bdt
import datasets.ss_transforms as sstr
from config.enums import DatasetOptions, ExperimentPhase, ModelOptions, OptimizerOptions, SchedulerOptions
from config.args import get_parser
from datasets.impl_factories import GTADatasetFactory, IddaDatasetFactory, IddaDatasetSelfLearningFactory, TransformsFactory
from experiment.impl_factories import CentralizedFactory, FederatedFactory, FederatedSelfLearningFactory, SiloLearningFactory
from models.abs_factories import OptimizerFactory, SchedulerFactory
from models.impl_factories import AdamFactory, \
                                  DeepLabV3MobileNetV2Factory, \
                                  LambdaSchedulerFactory, \
                                  SGDFactory, \
                                  StepLRSchedulerFactory
from utils.stream_metrics import StreamSegMetrics
from utils.utils import HardNegativeMining, MeanReduction
from loggers.logger import DummyLogger, LocalLoggerDecorator, Logger, WandbLoggerDecorator

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_metrics(args, train_ds: bdt.BaseDataset, targer_ds: bdt.BaseDataset):
    num_classes_train = train_ds.get_classes_number()
    num_classes_test = targer_ds.get_classes_number()
    if args.model == ModelOptions.DEEPLABv3_MOBILENETv2:
        metrics = {
            'source_train': StreamSegMetrics(num_classes_train, 'source_train'),
            'target_train': StreamSegMetrics(num_classes_test, 'target_train'),
            'test_same_dom': StreamSegMetrics(num_classes_test, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes_test, 'test_diff_dom')
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
    -> Tuple[List[bdt.BaseDataset], List[bdt.BaseDataset]]:
    print('Generating Datasets... \U0001F975')
    training_datasets = None
    test_datasets = None
    idda_factory = IddaDatasetFactory(args.framework, train_transforms, test_transforms)
    gta_factory = GTADatasetFactory(train_transforms)
    idda_sl_factory = IddaDatasetSelfLearningFactory(args.framework, train_transforms, None)
    match args.training_ds:
        case DatasetOptions.IDDA:
            training_datasets = idda_factory.construct_trainig_dataset()
        case DatasetOptions.GTA:
            training_datasets = gta_factory.construct_trainig_dataset()
        case DatasetOptions.IDDA_SELF:
            training_datasets = idda_sl_factory.construct_trainig_dataset()
        case _:
            raise NotImplementedError("The dataset chosen for training is not implemented")
        
    match args.test_ds:
        case DatasetOptions.IDDA:
            match args.training_ds:
                case DatasetOptions.IDDA:
                    test_datasets = idda_factory.construct_test_dataset()
                case DatasetOptions.GTA | DatasetOptions.IDDA_SELF:
                    idda_factory.set_in_test_mode()
                    test_datasets = idda_factory.construct_test_dataset()
        case _:
            raise NotImplementedError("The dataset chosen for training is not implemented")
    return training_datasets, test_datasets
        
def get_model(args: Namespace):
    print(f'Initializing Model... \U0001F975')
    match args.model:
        case ModelOptions.DEEPLABv3_MOBILENETv2:
            return DeepLabV3MobileNetV2Factory(dataset_type=args.training_ds).construct()
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

def main():
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
        metrics = set_metrics(args, train_datasets[0], test_datasets[0])
        
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
            case 'self_learning':
                experiment = FederatedSelfLearningFactory(args=args, 
                                 train_datasets=train_datasets, 
                                 test_datasets=test_datasets, 
                                 model=model, 
                                 metrics=metrics, 
                                 reduction=reduction, 
                                 optimizer_factory=optimizer_factory, 
                                 scheduler_factory=scheduler_factory,
                                 logger=logger).construct()
            case 'silo_self_learning':
                experiment = SiloLearningFactory(args=args, 
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

        starting = 0
        if args.load_checkpoint is not None:
            snapshot = logger.restore_snapshot(*args.load_checkpoint)
            if snapshot is not None:
                starting = experiment.load_snapshot(snapshot)

        match args.phase:
            case ExperimentPhase.ALL:
                experiment.train(starting)
                snapshot = experiment.save()
                logger.save(snapshot)
                experiment.eval_train()
                experiment.test()
            case ExperimentPhase.TRAIN:
                experiment.train(starting)
                snapshot = experiment.save()
                logger.save(snapshot)
            case ExperimentPhase.TEST:
                experiment.eval_train()
                experiment.test()
            case _:
                raise NotImplementedError("The phase chosen is not implemented")

        end = time.time()
        print(f"Elapsed time: {round(end - start, 2)}")
    finally:
        logger.finish()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()