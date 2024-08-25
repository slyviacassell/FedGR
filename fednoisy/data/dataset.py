import re
import torch
import argparse
import sys
import os
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
from PIL import Image
from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork

from fedlab.contrib.dataset.basic_dataset import FedDataset
from fedlab.utils.logger import Logger

from fednoisy.data.NLLData import functional as nllF

from fednoisy.data.NLLData.functional import NoisyDataset
from fednoisy.data import TRAIN_TRANSFORM_STRONG, TEST_TRANSFORM
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)


class FedNLLDataset(FedDataset):
    """Get dataset FedNLL setting from local files, can directly generate DataLoader
    used for train/test.

    Args:
        args:
        train_preload (bool): whether to preload train data into memory
        test_preload (bool): whether to preload test data into memory


    """

    def __init__(self, args, train_preload=False, test_preload=False) -> None:
        nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(args.data_dir, nll_filename)
        self.get_metadata(nll_file_path)
        self.nll_folder = os.path.join(args.data_dir, f"{nll_name}_seed_{args.seed}")
        self.train_preload = train_preload
        self.test_preload = test_preload
        if train_preload:
            self.train_datasets = {cid: None for cid in range(args.num_clients)}
            for cid in range(args.num_clients):
                self.train_datasets[cid] = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
                self.train_datasets[cid].legacy_noisy_labels = deepcopy(self.train_datasets[cid].noisy_labels)
            print(f"Client train datasets preloaded.")
        if test_preload:
            self.test_dataset = torch.load(
                os.path.join(self.nll_folder, "test-data.pkl")
            )
            print(f"Test datasets preloaded.")

        self.num = TRAIN_SAMPLE_NUM[args.dataset]
        self.dataset_name = args.dataset
        self.test_loader = None

        self.loader_cache = False
        self.train_queue_size = 20
        self.train_p = 0 # queue pointer
        self.train_p_map = {}
        self.train_cid_map = [None] * self.train_queue_size
        self.train_loaders_queue = [None] * self.train_queue_size # loader queue
        
        if self.train_preload:
            self.overall_noisy_ratio = self.get_overall_noisy_ratio() # not useful for server handler
    
    def dequeue_enqueue(self, loader, cid, queue_p: int, queue: list, p_map: dict, cid_map: list, queue_size: int):
        # dequeue
        dequeue_cid = cid_map[queue_p]
        if dequeue_cid is not None:
            p_map.pop(dequeue_cid)

        # enqueue
        queue[queue_p] = loader
        cid_map[queue_p] = cid
        p_map[cid] = queue_p
        return (queue_p + 1) % queue_size

    def get_dataset(self, cid=None, train=True):
        if train:
            if self.train_preload:
                dataset = self.train_datasets[cid]
            else:
                dataset = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
        else:
            if self.test_preload:
                dataset = self.test_dataset
            else:
                dataset = torch.load(os.path.join(self.nll_folder, "test-data.pkl"))
        return dataset
    
    def get_metadata(self, nll_file_path):
        self.metadata = torch.load(nll_file_path)

    def get_overall_noisy_ratio(self):
        dataset = OverallDataset(self.train_datasets)
        overall_noisy_ratio = np.sum(np.array(dataset.noisy_labels) != np.array(dataset.labels))
        overall_noisy_ratio /= len(dataset)
        return overall_noisy_ratio

    def get_dataloader(self, cid=None, train=True, batch_size=64, num_workers=2):
        
        if train:
            if self.train_p_map.get(cid, None) is None: # speedup data loader init, more memory usage
                dataset = self.get_dataset(cid, train)
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                )
                if self.loader_cache:
                    self.train_p = self.dequeue_enqueue(data_loader,cid,self.train_p,self.train_loaders_queue,self.train_p_map,self.train_cid_map,self.train_queue_size)
            else:
                data_loader = self.train_loaders_queue[self.train_p_map[cid]]
        else:
            if self.test_loader is None:
                dataset = self.get_dataset(cid, train)
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                )
                self.test_loader = data_loader
            else:
                data_loader = self.test_loader

        return data_loader
    
    def get_eval_train_dataloader(self, dataset_name, cid=None, batch_size=128, num_workers=2):
        dataset = self.get_dataset(cid, train=True)
        dataset = deepcopy(dataset)
        dataset.transform = TEST_TRANSFORM[dataset_name] # replace transform

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return data_loader
    
    def get_overall_dataloader(self,batch_size=64,num_workers=2):
        dataset = OverallDataset(self.train_datasets)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
    
    def get_noisy_discrepancy(self, cid=None):
        dataset = self.get_dataset(cid)
        label_distri = np.bincount(dataset.noisy_labels, minlength=CLASS_NUM[self.dataset_name])
        label_distri = label_distri / np.sum(label_distri) + 1e-8
        ideal_distri = np.ones(CLASS_NUM[self.dataset_name]) / CLASS_NUM[self.dataset_name]
        
        # js div
        # kl1 = (ideal_distri * (np.log(ideal_distri) - np.log((label_distri+ideal_distri)/2))).sum()
        # kl2 = (label_distri * (np.log(label_distri) - np.log((label_distri+ideal_distri)/2))).sum()
        # js_div = (kl1 + kl2) / 2
        # return js_div

        # kl div
        kl = (label_distri * (np.log(label_distri) - np.log(ideal_distri))).sum()
        return kl
    
    # def update_labels(self, p_labels: pd.DataFrame, noisy_guids: np.ndarray):
    #     for cid,d in self.train_datasets.items():
    #         for i, guid in enumerate(d.guids):
    #             if guid in noisy_guids:
    #                 d.noisy_labels[i] = p_labels.loc[guid]["freqent_pred"]
    #         c_mask = np.array(d.legacy_noisy_labels) == np.array(d.labels)
    #         pc_mask = np.array(d.noisy_labels) == np.array(d.labels)
    #         print(
    #             f"update Client-{cid} {c_mask.sum() / len(c_mask)*100:.2f}% -> {pc_mask.sum() / len(pc_mask)*100:.2f}% "
    #             f"({pc_mask[c_mask].sum() / len(pc_mask)*100:.2f}%, {pc_mask[~c_mask].sum() / len(pc_mask)*100:.2f}%)"
    
    #         )

    def get_dividemix_dataloader(self, cid=None, train=True, batch_size=64, num_workers=2, selected_guid: np.ndarray=None, sample_prob: Dict=None):
        if train:
            shuffle = True
        else:
            shuffle = False

        dataset = self.get_dataset(cid, train)
        dataset = DivideMixFedNLLDataset(dataset)
        if selected_guid is not None and sample_prob is not None:
            dataset.update(selected_guid,sample_prob)

        if len(dataset) == 0:
            print('f@cked',selected_guid)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True, 
            persistent_workers=True,
        )
        return data_loader
    
    def get_semiws_dataloader(self, cid=None, train=True, batch_size=64, num_workers=2, selected_guid: np.ndarray=None, prob_dict: Dict=None):
        if train:
            shuffle = True
        else:
            shuffle = False

        dataset = self.get_dataset(cid, train)
        dataset = SemiWSFedNLLDataset(self.dataset_name,dataset)
        if selected_guid is not None:
            dataset.update(selected_guid, prob_dict)

        if len(dataset) == 0:
            print('fucked',selected_guid)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # pin_memory=True, # memory leak for fedlab scale
            persistent_workers=True,
        )
        return data_loader


class DivideMixFedNLLDataset(Dataset):
    def __init__(self, dataset: NoisyDataset, aug_times: int=2) -> None:
        super().__init__()
        self.dataset = dataset
        self.aug_times = aug_times
        self.pseudo_labels = None
        self.sample_reset()

    def __getitem__(self, index: Any) -> Any:
        if self.dataset.folder_data is False:
            # CIFAR10 & CIFAR100 & SVHN & MNIST
            img, label = self.data[index], self.labels[index]
            img = Image.fromarray(img, mode=self.dataset.mode)
        else:
            # clothing1m data
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path).convert("RGB")
        guid = self.guids[index]

        if self.dataset.train:
            noisy_label = self.noisy_labels[index]
            weight = self.sample_weight[index]
            if self.aug_times == 1:
                img = self.dataset.transform(img)
                sample = {"img": img}
            else:
                img = [self.dataset.transform(img) for _ in range(self.aug_times)]
                sample = {f"img_{i}": img[i] for i in range(self.aug_times)}
            
            if self.pseudo_labels is not None:
                pesudo_label = self.pseudo_labels[index]
                sample.update(
                    {
                        "pesudo_label": pesudo_label,
                    }
                )

            sample.update(
                {
                    "label": label, 
                    "noisy_label": noisy_label,
                    "guid": guid,
                    "weight": weight,
                }
            )
            return sample
        else:
            img = self.dataset.transform(img)
            return {
                "img": img, 
                "label": label,
                "guid": guid,
            }
    
    def update(self, selected_guid: np.ndarray, sample_weight: Dict = None, pseudo_labels: Dict = None):
        selected_idx = self.get_selected_idx(selected_guid)
        self.data = [self.dataset.data[i] for i in selected_idx]
        self.guids = [self.dataset.guids[i] for i in selected_idx]
        self.noisy_labels = [self.dataset.noisy_labels[i] for i in selected_idx]
        self.labels = [self.dataset.labels[i] for i in selected_idx]
        if pseudo_labels is not None:
            self.pseudo_labels = [pseudo_labels[guid] for guid in self.guids]

        if sample_weight is not None:
            self.sample_weight = np.array([sample_weight[guid] for guid in self.guids])

    def get_selected_idx(self, selected_guid: np.ndarray):
        idx = [i for i, guid in enumerate(self.dataset.guids) if guid in selected_guid]
        return idx
    
    def sample_reset(self):
        self.data = self.dataset.data
        self.labels = self.dataset.labels
        self.guids = self.dataset.guids
        self.noisy_labels = self.dataset.noisy_labels
        self.sample_weight = np.ones(len(self))
    
    def __len__(self):
        return len(self.labels)
    

class SemiWSFedNLLDataset(DivideMixFedNLLDataset):
    def __init__(self, dataset_name:str, dataset: NoisyDataset, aug_times: int = 2) -> None:
        super().__init__(dataset, aug_times)
        self.weak_transform = self.dataset.transform
        self.strong_transform = TRAIN_TRANSFORM_STRONG[dataset_name]
        self.pseudo_labels = None

    def __getitem__(self, index: Any) -> Any:
        if self.dataset.folder_data is False:
            # CIFAR10 & CIFAR100 & SVHN & MNIST
            img, label = self.data[index], self.labels[index]
            img = Image.fromarray(img, mode=self.dataset.mode)
        else:
            # clothing1m data
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path).convert("RGB")
        guid = self.guids[index]

        if self.dataset.train:
            noisy_label = self.noisy_labels[index]
            sample_weight = self.sample_weight[index]
            sample = {
                "img_w": self.weak_transform(img),
                "img_s": self.strong_transform(img),
            }

            if self.pseudo_labels is not None:
                pesudo_label = self.pseudo_labels[index]
                sample.update(
                    {
                        "pesudo_label": pesudo_label,
                    }
                )

            sample.update(
                {
                    "label": label, 
                    "noisy_label": noisy_label,
                    "guid": guid,
                    "weight": sample_weight,
                }
            )
            return sample
        else:
            img = self.dataset.transform(img)
            return {
                "img": img, 
                "label": label,
                "guid": guid,
            }
        
    def update(self, selected_guid: np.ndarray, sample_weight: Dict = None, pseudo_labels: Dict = None):
        selected_idx = self.get_selected_idx(selected_guid)
        self.data = [self.dataset.data[i] for i in selected_idx]
        self.guids = [self.dataset.guids[i] for i in selected_idx]
        self.noisy_labels = [self.dataset.noisy_labels[i] for i in selected_idx]
        self.labels = [self.dataset.labels[i] for i in selected_idx]
        if pseudo_labels is not None:
            self.pseudo_labels = [pseudo_labels[guid] for guid in self.guids]
        
        if sample_weight is not None:
            self.sample_weight = np.array([sample_weight[guid] for guid in self.guids])
    
    def sample_reset(self):
        self.data = self.dataset.data
        self.labels = self.dataset.labels
        self.guids = self.dataset.guids
        self.noisy_labels = self.dataset.noisy_labels
        self.sample_weight = np.ones(len(self))


class OverallDataset(Dataset):
    def __init__(self, cid_datasets: Dict) -> None:
        super().__init__()
        self.train_datasets = cid_datasets
        self.concat_datasets()
        self.folder_data = self.train_datasets[0].folder_data
        if self.folder_data is False:
            self.mode = self.train_datasets[0].mode
        self.transform = self.train_datasets[0].transform
        self.train = self.train_datasets[0].train

    def concat_datasets(self):
        self.data = []
        self.labels = []
        self.guids = []
        self.noisy_labels = []
        for cid in self.train_datasets:
            assert isinstance(self.train_datasets[cid].data, list)
            assert isinstance(self.train_datasets[cid].labels, list)
            assert isinstance(self.train_datasets[cid].guids, list)
            assert isinstance(self.train_datasets[cid].noisy_labels, list)
            self.data += self.train_datasets[cid].data
            self.labels += self.train_datasets[cid].labels
            self.guids += self.train_datasets[cid].guids
            self.noisy_labels += self.train_datasets[cid].noisy_labels

    def __getitem__(self, index: Any) -> Any:
        if self.folder_data is False:
            # CIFAR10 & CIFAR100 & SVHN & MNIST
            img, label = self.data[index], self.labels[index]
            img = Image.fromarray(img, mode=self.mode)
        else:
            # clothing1m data
            img_path, label = self.data[index], self.labels[index]
            img = Image.open(img_path).convert("RGB")
        guid = self.guids[index]

        if self.train:
            noisy_label = self.noisy_labels[index]

            sample = {
                "img": self.transform(img),
            }

            sample.update(
                {
                    "label": label, 
                    "noisy_label": noisy_label,
                    "guid": guid,
                }
            )
            return sample
        else:
            img = self.transform(img)
            return {
                "img": img, 
                "label": label,
                "guid": guid,
            }
        
    def __len__(self):
        return len(self.labels)
    