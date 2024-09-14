import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fedlab.utils.dataset import functional as partF
from fedlab.utils.dataset.partition import DataPartitioner, VisionPartitioner

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    TRANSITION_MATRIX,
    NORM_VALUES,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)
from fednoisy import visual
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class FedNLLScene(NLLBase):
    centralized = False

    def __init__(
        self,
        root_dir: str,
        out_dir: str,
        num_clients: int,
        globalize: bool,
        noise_mode: str = "clean",
        partition: str = "iid",
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        noise_ratio: float = 0.0,
        min_noise_ratio: float = 0,
        max_noise_ratio: float = 1.0,
        min_require_size: Optional[int] = 10,
        partitioner: Optional[DataPartitioner] = VisionPartitioner,
        local_noise_mode: str = "uniform",
        noise_ratio_mu: float = 0.0,
        noise_ratio_sigma: float = 0.0,
        noisy_client_ratio: float = 1.0,
    ) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)

        self.num_clients = num_clients
        if partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            assert isinstance(major_classes_num, int), (
                f"'major_classes_num' should be integer, "
                f"not {type(major_classes_num)}."
            )
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, (
                f"'major_classes_num' for each client "
                f"should be less than number of total "
                f"classes {self.num_classes}."
            )

        elif partition in ["noniid-labeldir", "noniid-quantity"]:
            # label-distribution-skew(Dirichlet) and quantity-distribution-skew (Dirichlet)
            assert dir_alpha > 0, (
                f"Parameter 'dir_alpha' for Dirichlet distribution should be "
                f"positive."
            )

        elif partition == "iid":
            pass
        else:
            raise ValueError(
                f"Data partition only supports 'noniid-labeldir', 'noniid-quantity', 'iid'. "
                f"{partition} is not supported."
            )

        self.partition = partition

        if globalize is True:
            noise_ratio = F.check_centr_noisy_setting(
                noise_mode, noise_ratio
            )  # check validation of noise setting
            self.noise_ratio = {cid: noise_ratio for cid in range(num_clients)}
            self.local_noise_mode = None # used for local label noise
        else:
            self.local_noise_mode = local_noise_mode
            if local_noise_mode == "uniform":
                self.noise_ratio = {cid: 0.0 for cid in range(num_clients)}  # initial value
                self.min_noise_ratio = min_noise_ratio
                self.max_noise_ratio = max_noise_ratio
            elif local_noise_mode == "gaussian":
                self.noise_ratio = {cid: 0.0 for cid in range(num_clients)}
                self.noise_ratio_mu = noise_ratio_mu
                self.noise_ratio_sigma = noise_ratio_sigma
            else:
                raise ValueError(
                    f"Local noise mode only supports 'uniform' and 'gaussian'. "
                    f"{local_noise_mode} is not supported."
                )

        self.globalize = globalize
        self.min_require_size = min_require_size
        self.dir_alpha = dir_alpha
        self.major_classes_num = major_classes_num
        self.partitioner = partitioner
        self.noisy_client_raio = min(noisy_client_ratio, 1.0)

    def create_nll_scene(self, seed: int = 0):
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{seed}_setting.pt"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        self.nll_scene_folder = os.path.join(self.out_dir, f"{self}_seed_{seed}")
        client_dict = self._perform_partition() # partition with true labels
        for cid in client_dict:
            client_dict[cid] = client_dict[cid].tolist()  # change array([...]) to [...]
        self.client_dict = client_dict
        self.data_dict = F.split_data(self.client_dict, self.train_data)
        self.labels_dict = F.split_data(self.client_dict, self.train_labels)
        if hasattr(self,"train_guids"): # dataset cartography
            self.guids_dict = F.split_data(self.client_dict, self.train_guids) 
        else:
            self.guids_dict = None
        self.noisy_labels_dict, self.transition_matrix_dict = self._gen_noisy_labels(client_dict)
        self.true_noise_ratio = F.cal_multiple_true_noisy_ratio(
            self.labels_dict, self.noisy_labels_dict
        )
        print(f"True noisy ratio is calculated: {self.true_noise_ratio}")

    def save_nll_scene(self):
        fednll_scene = {
            "dataset": self.dataset_name,
            "client_dict": self.client_dict,
            "partition": self.partition,
            "dir_alpha": self.dir_alpha,
            "num_clients": self.num_clients,
            "major_classes_num": self.major_classes_num,
            "min_require_size": self.min_require_size,
            "globalize": self.globalize,
            "noise_mode": self.noise_mode,
            "true_noise_ratio": self.true_noise_ratio,
            "noise_ratio": self.noise_ratio,
            "noisy_labels": self.noisy_labels_dict,
            "transition_matrix_dict": self.transition_matrix_dict,
            "local_noise_mode": self.local_noise_mode,
            "overall_noise_ratio": self.overall_noise_ratio,
            "client_noise_mode": self.cid_noise_mode,
        }

        torch.save(fednll_scene, self.nll_scene_file_path)
        print(
            f"Federated Noisy Label Learning scene saved to {self.nll_scene_file_path}, with keys "
            f"{list(fednll_scene.keys())}"
        )

        os.mkdir(self.nll_scene_folder)
        # train split save to local
        train_transform = TRAIN_TRANSFORM[self.dataset_name]
        for cid in range(self.num_clients):
            client_dataset = NoisyDataset(
                data=self.data_dict[cid],
                labels=self.labels_dict[cid],
                guids=self.guids_dict[cid] if self.guids_dict is not None else self.guids_dict,
                noisy_labels=self.noisy_labels_dict[cid],
                train=True,
                transform=train_transform,
            )
            path = os.path.join(self.nll_scene_folder, f"train-data{cid}.pkl")
            torch.save(client_dataset, path)
            print(f"Client {cid} local train set saved to {path}")

        # test split save to local
        test_transform = TEST_TRANSFORM[self.dataset_name]
        test_dataset = NoisyDataset(
            data=self.test_data,
            labels=self.test_labels,
            guids=self.test_guids,
            train=False,
            transform=test_transform,
        )
        path = os.path.join(self.nll_scene_folder, "test-data.pkl")
        torch.save(test_dataset, path)
        print(f"Test set saved to {path}")

    def _perform_partition(self):
        if self.partition == "noniid-quantity":
            partition = "unbalance"
        else:
            partition = self.partition

        partitioner = self.partitioner(
            targets=np.array(self.train_labels),
            num_clients=self.num_clients,
            partition=partition,
            dir_alpha=self.dir_alpha,
            major_classes_num=self.major_classes_num,
            verbose=False,
            seed=self.seed,
        )

        return partitioner.client_dict

    def _gen_noisy_labels(self, client_dict):
        if os.path.exists(self.nll_scene_file_path):
            # read from local file if exists
            print(
                f"Federated noisy label learning scene {self}_seed_{self.seed} are already generated, "
                f"loaded from {self.nll_scene_file_path}."
            )
            entry = torch.load(self.nll_scene_file_path)
            noisy_labels_dict = entry["noisy_labels"]
            transition_matrix_dict = entry.get("transition_matrix_dict",{})
            self.overall_noise_ratio = entry["overall_noise_ratio"]

        else:
            class_space = set(self.train_labels)
            transition_matrix_dict = {}
            overall_noise_cnt = 0
            # generate noisy file
            if self.globalize is True:
                # todo: add specified noise user and mixed noise
                self.cid_noise_mode = {cid: self.noise_mode for cid in range(self.num_clients)}

                # globalized FedNLL
                noisy_labels = F.generate_noisy_labels(
                    labels=self.train_labels,
                    noise_mode=self.noise_mode,
                    noise_ratio=self.noise_ratio[0],
                    transition=TRANSITION_MATRIX[self.dataset_name],
                    num_classes=self.num_classes,
                )
                noisy_labels_dict = F.split_data(client_dict, noisy_labels)

                transition_matrix = F.get_transition_matrix(self.train_labels, noisy_labels, class_space)
                for cid in range(self.num_clients):
                    transition_matrix_dict[cid] = transition_matrix

                self.overall_noise_ratio = self.noise_ratio[0]
            else:
                # localized FedNLL
                noisy_client_num = int(self.num_clients * self.noisy_client_raio)
                noisy_client_idx = np.random.permutation(self.num_clients)
                if noisy_client_num < self.num_clients:
                    noisy_client_idx = noisy_client_idx[:noisy_client_num]
                
                if self.noise_mode == "mixed":
                    self.cid_noise_mode = {cid: random.choice(["sym","asym"]) if cid in noisy_client_idx else "clean" for cid in range(self.num_clients)}
                else:
                    self.cid_noise_mode = {cid: self.noise_mode if cid in noisy_client_idx else "clean" for cid in range(self.num_clients)}

                noisy_labels_dict = dict()
                if self.local_noise_mode == "gaussian":
                    self.noise_ratio = {
                        cid: np.random.normal(self.noise_ratio_mu, self.noise_ratio_sigma) if cid in noisy_client_idx else 0.0
                        for cid in range(self.num_clients)
                    }
                elif self.local_noise_mode == "uniform":
                    self.noise_ratio = {
                        cid: np.random.uniform(self.min_noise_ratio, self.max_noise_ratio) if cid in noisy_client_idx else 0.0
                        for cid in range(self.num_clients)
                    }
                print(f"Current noise ratios: {self.noise_ratio}")
                print(f"Current noise mode: {self.cid_noise_mode}")
                for cid in range(self.num_clients):
                    cur_labels = self.labels_dict[cid]
                    # cur_noisy_labels = F.generate_local_noisy_labels( # based on the local labels
                    #     cur_labels,
                    #     noise_mode=self.noise_mode,
                    #     noise_ratio=self.noise_ratio[cid],
                    #     transition=TRANSITION_MATRIX[self.dataset_name],
                    #     dataset=self.dataset_name,
                    # )
                    cur_noisy_labels = F.generate_local_noisy_labels_class_space( # based on the global class space
                        cur_labels,
                        class_space,
                        noise_mode=self.cid_noise_mode[cid],
                        noise_ratio=self.noise_ratio[cid],
                        transition=TRANSITION_MATRIX[self.dataset_name],
                        dataset=self.dataset_name,
                    )
                    noisy_labels_dict[cid] = cur_noisy_labels

                    overall_noise_cnt += np.sum(np.array(cur_labels) != np.array(cur_noisy_labels))

                    cur_transition_matrix = F.get_transition_matrix(cur_labels, cur_noisy_labels, class_space)
                    transition_matrix_dict[cid] = cur_transition_matrix
                self.overall_noise_ratio = overall_noise_cnt / len(self.train_labels)
            print(
                f"Federated noisy label learning scene {self}_seed_{self.seed} are generated."
            )
        return noisy_labels_dict, transition_matrix_dict

    @property
    def partition_setting(self):
        if self.partition == "noniid-#label":
            partition_param = f"{self.major_classes_num}"
        elif self.partition == "noniid-quantity":
            partition_param = f"{self.dir_alpha}"
        elif self.partition == "noniid-labeldir":
            partition_param = f"{self.dir_alpha:.2f}_{self.min_require_size}"
        else:
            # IID
            partition_param = ""
        return f"{self.num_clients}_{self.partition}_{partition_param}"

    @property
    def noise_setting(self):
        if self.globalize is False:
            if self.local_noise_mode == "uniform":
                noise_param = f"local_{self.noise_mode}_min_{self.min_noise_ratio:.2f}_max_{self.max_noise_ratio:.2f}"
            elif self.local_noise_mode == "gaussian":
                noise_param = f"local_{self.local_noise_mode}_{self.noise_mode}_mu_{self.noise_ratio_mu:.2f}_sigma_{self.noise_ratio_sigma:.2f}"

            if self.noisy_client_raio < 1.0:
                noise_param += f"_noisy_{self.noisy_client_raio:.2f}"
        else:
            noise_param = f"global_{self.noise_mode}_{self.noise_ratio[0]:.2f}"
        return noise_param

    @property
    def setting(self):
        return f"{self.partition_setting}_{self.noise_setting}"
