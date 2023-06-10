import json
import os
from argparse import Namespace
from typing import List, Optional, Tuple
from overrides import override
from torch.utils.data import DataLoader
from datasets.silo_idda import SiloIddaDataset
from datasets.sl_idda import IDDADatasetSelfLearning

import datasets.ss_transforms as sstr
from config.enums import NormOptions
from datasets.abs_factories import DatasetFactory
from datasets.base_dataset import BaseDataset
from datasets.gta import GTADataset
from datasets.idda import IDDADataset

class IddaDatasetFactory(DatasetFactory):

    def __init__(self,
                 framework: str,
                 train_transforms: Optional[sstr.Compose],
                 test_transforms: Optional[sstr.Compose],
                 test_dataset: bool = False) -> None:
        super().__init__(root="data/idda", 
                         train_transforms=train_transforms, 
                         test_transforms=test_transforms)
        self.framework = framework
        self.test_dataset = test_dataset

    @override
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        train_datasets = []
        match self.framework:
            case "centralized":
                with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                    all_data = f.readlines()
                    train_datasets.append(IDDADataset(root=self.root,
                                    list_samples=all_data,
                                    transform=self.train_transforms,
                                    test_mode=False,
                                    client_name="train"))
            case "federated":
                    with open(os.path.join(self.root, 'train.json'), 'r') as f:
                        all_data = json.load(f)
                        for client_id in all_data.keys():
                                train_datasets.append(IDDADataset(root=self.root, 
                                                            list_samples=all_data[client_id], 
                                                            transform=self.train_transforms,
                                                            client_name=client_id))
            case _:
                raise NotImplementedError(f"IDDA training dataset is not implemented for {self.framework}")
        return train_datasets

    @override
    def construct_test_dataset(self) -> List[BaseDataset]:
        test_datasets= []
        if self.test_dataset == True:
            with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                all_data = f.readlines()
                test_datasets.append(IDDADataset(root=self.root,
                                        list_samples=all_data,
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name="target_train"))
        with open(os.path.join(self.root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_datasets.append(IDDADataset(root=self.root,
                                        list_samples=test_same_dom_data, 
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name='test_same_dom'))
        with open(os.path.join(self.root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_datasets.append(IDDADataset(root=self.root,
                                        list_samples=test_diff_dom_data,
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name='test_diff_dom'))
        return test_datasets
    
    def set_in_test_mode(self) -> None:
        self.test_dataset = True

    
class GTADatasetFactory(DatasetFactory):

    def __init__(self,
                 train_transforms: sstr.Compose,) -> None:
        super().__init__("data/gta", train_transforms, None)

    @override    
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        with open(os.path.join(self.root, 'train.txt'), 'r') as f:
            all_data = f.readlines()
            return [GTADataset(root=self.root,
                               list_samples=all_data,
                               transform=self.train_transforms,
                               client_name="train")]

    @override
    def construct_test_dataset(self) -> List[BaseDataset]:
        raise NotImplementedError("Test set for GTA dataset is not implemented")
    
class IddaDatasetSelfLearningFactory(DatasetFactory):

    def __init__(self,
                 framework: str,
                 train_transforms: Optional[sstr.Compose],
                 test_transforms: Optional[sstr.Compose]) -> None:
        super().__init__(root="data/idda", 
                         train_transforms=train_transforms, 
                         test_transforms=test_transforms)
        self.framework = framework
    
    @override
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        train_datasets = []
        match self.framework:
            case "centralized" | "federated" | "silo_self_learning":
                raise NotImplementedError("NO")
            case "self_learning":
                    with open(os.path.join(self.root, 'train.json'), 'r') as f:
                        all_data = json.load(f)
                        for client_id in all_data.keys():
                                train_datasets.append(IDDADatasetSelfLearning(root=self.root, 
                                                            list_samples=all_data[client_id], 
                                                            transform=self.train_transforms,
                                                            client_name=client_id))
        return train_datasets
    
    @override
    def construct_test_dataset(self) -> List[BaseDataset]:
        raise NotImplementedError("Test set for Idda Self Learning dataset is not implemented")
    
class SiloIddaDatasetFactory(IddaDatasetFactory):

    def __init__(self, 
                 framework: str, 
                 train_transforms: sstr.Compose | None, 
                 test_transforms: sstr.Compose | None, 
                 test_dataset: bool = False) -> None:
        super().__init__(framework, 
                         train_transforms, 
                         test_transforms, 
                         test_dataset)
    
    @override
    def construct_trainig_dataset(self) -> List[BaseDataset]:
        train_datasets = []
        match self.framework:
            case "centralized" | "federated" | "self_learning":
                raise NotImplementedError("NO")
            case "silo_self_learning":
                    with open(os.path.join(self.root, 'train.json'), 'r') as f:
                        all_data = json.load(f)
                        for client_id in all_data.keys():
                                train_datasets.append(SiloIddaDataset(root=self.root, 
                                                                      list_samples=all_data[client_id], 
                                                                      transform=self.train_transforms,
                                                                      client_name=client_id))
        return train_datasets
    
    @override
    def construct_test_dataset(self) -> List[BaseDataset]:
        test_datasets= []
        if self.test_dataset == True:
            with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                all_data = f.readlines()
                test_datasets.append(SiloIddaDataset(root=self.root,
                                        list_samples=all_data,
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name="target_train"))
        with open(os.path.join(self.root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_datasets.append(SiloIddaDataset(root=self.root,
                                        list_samples=test_same_dom_data, 
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name='test_same_dom'))
        with open(os.path.join(self.root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_datasets.append(SiloIddaDataset(root=self.root,
                                        list_samples=test_diff_dom_data,
                                        transform=self.test_transforms,
                                        test_mode=True,
                                        client_name='test_diff_dom'))
        return test_datasets
        
class TransformsFactory():

    def __init__(self, args: Namespace) -> None:
        self.rsrc_transform = args.rsrc_transform
        self.rrc_transform = args.rrc_transform
        self.jitter = args.jitter
        self.h_resize = args.h_resize
        self.w_resize = args.w_resize
        self.min_scale = args.min_scale
        self.max_scale = args.max_scale
        self.use_fda = args.fda
        self.fda_L = args.fda_L
        match args.norm:
            case NormOptions.EROS:
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
            case NormOptions.CTS:
                self.mean = [0.5, 0.5, 0.5]
                self.std = [0.5, 0.5, 0.5]
            case NormOptions.GTA:
                self.mean = [73.158359210711552, 82.908917542625858, 72.392398761941593]
                self.std =  [47.675755341814678, 48.494214368814916, 47.736546325441594]

    def construct(self) -> Tuple[sstr.Compose, sstr.Compose]:
        train_transform = []
        test_transform = []

        if self.use_fda:
            dss = IddaDatasetFactory("federated", sstr.ToTensor(), None).construct_trainig_dataset()
            loaders = []
            for ds in dss:
                loaders.append(DataLoader(ds))
            train_transform.append(sstr.FDA(loaders, self.fda_L))

        train_transform.append(sstr.RandomHorizontalFlip(0.5))
        
        if self.rsrc_transform:
            train_transform.append(
                sstr.RandomScaleRandomCrop(crop_size=(1024, 1856), scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
            train_transform.append(sstr.Resize(size=(self.h_resize, self.w_resize)))

        elif self.rrc_transform:
            train_transform.append(
                sstr.RandomResizedCrop((self.h_resize, self.w_resize), scale=(self.min_scale, self.max_scale)))
    
        if self.jitter:
            train_transform.append(sstr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
    
        train_transform = train_transform + [sstr.ToTensor(), sstr.Normalize(mean=self.mean, std=self.std)]
        train_transform = sstr.Compose(train_transform)

        test_transform = test_transform + [sstr.ToTensor(), sstr.Normalize(mean=self.mean, std=self.std)]
        test_transform = sstr.Compose(test_transform)

        return train_transform, test_transform