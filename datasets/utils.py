from argparse import Namespace
import json
import os
from typing import Tuple
from datasets.idda import IDDADataset
import datasets.ss_transforms as sstr


def get_dataset_num_classes(dataset: str) -> int:
    """Return the number of classes present in the dataset
    Args:
        dataset (str): name of the dataset
    Raises:
        NotImplementedError: the dataset selected is not supported
    Returns:
        int: number of classes
    """    
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError

def get_transforms(args: Namespace) -> Tuple[sstr.Compose, sstr.Compose]:
    train_transform = []
    test_transform = []

    if args.eros_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_transform.append(sstr.RandomHorizontalFlip(0.5))
    
    if args.rsrc_transform:
        train_transform.append(
            sstr.RandomScaleRandomCrop(crop_size=(1024, 1856), scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
        train_transform.append(sstr.Resize(size=(args.h_resize, args.w_resize)))
    
    elif args.rrc_transform:
        train_transform.append(
            sstr.RandomResizedCrop((args.h_resize, args.w_resize), scale=(args.min_scale, args.max_scale)))
    
    if args.jitter:
        train_transform.append(sstr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
    
    train_transform = train_transform + [sstr.ToTensor(), sstr.Normalize(mean=mean, std=std)]
    train_transform = sstr.Compose(train_transform)

    if args.use_test_resize:
        test_transform.append(sstr.Resize(size=(512, 928)))
    test_transform = test_transform + [sstr.ToTensor(), sstr.Normalize(mean=mean, std=std)]
    test_transform = sstr.Compose(test_transform)

    return train_transform, test_transform

def get_datasets(args):
    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        if args.framework == "centralized":
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                all_data = f.readlines()
                train_datasets.append(IDDADataset(root=root, list_samples=all_data, transform=train_transforms,
                                                client_name=None))
        elif args.framework == "federated":
            with open(os.path.join(root, 'train.json'), 'r') as f:
                all_data = json.load(f)
            for client_id in all_data.keys():
                train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                                client_name=client_id))
        
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = {"same_dom": test_same_dom_dataset, 
                         "diff_dom": test_diff_dom_dataset}
    else:
        raise NotImplementedError

    return train_datasets, test_datasets