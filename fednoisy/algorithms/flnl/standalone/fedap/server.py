import sys
import argparse
import os
import random
import numpy as np

from typing import List
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as TF

from fedlab.core.server.manager import SynchronousServerManager

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, Aggregators, SerializationTool

sys.path.append(os.getcwd())
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)

from fednoisy.utils.misc import AverageMeter
from fednoisy.utils import misc as misc
from fednoisy.utils.wandb_logger import WandbLogger
from fednoisy.utils.ema import EMA

from fednoisy.core import SynServerAlogrithmBase
from fednoisy.core.hooks import (
    TestHook,
    GlobalGradNormMonitorHook,
)
from fednoisy.algorithms.flnl.hooks import (
    FedProxGlobalAdaptiveMuScheduler,
    FedProxMuConstantScheduler,
)


class FedAPServerHandler(SyncServerHandler, SynServerAlogrithmBase):
    def __init__( 
        self,
        model: torch.nn.Module,
        global_round: int,
        sample_ratio: float,
        nll_name: str = None,
        cuda: bool = True,
        device: str = None,
        logger: Logger = None,
        wandb_logger: WandbLogger=None,
        args=None,
    ):
        SyncServerHandler.__init__(
            self, model, global_round, sample_ratio, cuda, device, logger
        )
        SynServerAlogrithmBase.__init__(self)
        self.nll_name = nll_name
        self.args = args
        self.wandb_logger = wandb_logger

        self.set_hooks()

        self.on_init()

    def set_hooks(self):
        self.register_hooks(TestHook(), None, "LOWEST")
        self.register_hooks(GlobalGradNormMonitorHook(), None, "LOWEST")
        if self.args.use_fedprox:
            if self.args.fedprox_mu == "constant":
                self.register_hooks(FedProxMuConstantScheduler(self.args), "fedprox_mu_scheduler", "LOWEST")
            elif self.args.fedprox_mu == "adaptive":
                self.register_hooks(FedProxGlobalAdaptiveMuScheduler(self.args, mu_delta=0.01), "fedprox_mu_scheduler", "LOWEST")
        if type(self) ==  FedAPServerHandler:
            self._LOGGER.info(
                f"Server Registered hooks: {self.hooks_dict.keys()}"
            )

    def on_global_update_end(self, *args, **kwargs):
        super(FedAPServerHandler, self).on_global_update_end(*args, **kwargs)
        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                {
                    "other/round": self.round,
                },
                commit=True,
                step=self.round,
            ) # commit the log

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset

    def global_update(self, buffer):
        # the self.round increases after global updating
        parameters_list = [elem[0] for elem in buffer]
        weights = [elem[1] for elem in buffer]
        local_losses = [elem[2] for elem in buffer] # for fedprox adaptive mu scheduler
        cid_list = [elem[3].int().item() for elem in buffer]

        self.on_global_update_start()

        if self.args.use_fedprox:
            self.call_hook("step", "fedprox_mu_scheduler", local_losses, weights)
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        self.set_model(serialized_parameters)
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] server global update done. {cid_list}"
        )

        self.on_global_update_end()

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        if self.args.use_fedprox:
            mu = self.call_hook("get_mu", "fedprox_mu_scheduler")
        else:
            mu = 0.0
        return [self.model_parameters, mu]
    
    # fixme: unnormal shutdown
    # def sample_clients(self):        
    #     if self.num_clients_per_round < self.num_clients:
    #         # random sample the clients without replacements
    #         if self.round == 0:
    #             self.cnt = 0
    #             self.selected_set = list(range(self.num_clients))
    #             random.shuffle(self.selected_set)

    #         selection = self.selected_set[self.cnt*self.num_clients_per_round:(1+self.cnt)*self.num_clients_per_round]
                
    #         if len(self.selected_set) <= (1+self.cnt)*self.num_clients_per_round:
    #             self.selected_set = list(range(self.num_clients))
    #             random.shuffle(self.selected_set)
    #             self.cnt = 0
    #             self._LOGGER.info(f"Round [{self.round - 1}/{self.global_round}] server init client set for next")
    #         else:
    #             self.cnt += 1
    #     else:
    #         selection = random.sample(range(self.num_clients), self.num_clients_per_round)
        
    #     return sorted(selection)