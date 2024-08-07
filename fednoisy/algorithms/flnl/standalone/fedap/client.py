from ast import mod
from cgi import test
from re import L
import torch
import argparse
import sys
import os
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
import math
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from collections import Counter
from fednoisy import data
from fednoisy.data.NLLData.functional import NoisyDataset
from fednoisy.utils.ema import EMA

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as TF

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork

from fedlab.contrib.dataset.basic_dataset import FedDataset
from fedlab.utils.logger import Logger

sys.path.append(os.getcwd())
from fednoisy.data.NLLData import functional as nllF
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)
from fednoisy.utils.misc import (
    setup_seed,
    make_dirs,
    make_exp_name,
    result_parser,
    make_alg_name,
    AverageMeter,
)
from fednoisy.utils import misc as misc
from fednoisy.utils.criterion import get_robust_loss
from fednoisy.utils.mixup import mixup_data
from fednoisy.utils.lr_scheduler import get_lr_scheduler

from fednoisy.utils.wandb_logger import WandbLogger

from fednoisy.core import (
    SerialClientAlogrithmBase,
)
from fednoisy.core.hooks import (
    TestHook,
    EvaluateTrainHook,
    ClientGradClipHook,
    SerialClientLocalEMAHook,
    GlobalGradNormMonitorHook,
)


class FedAPClientTrainer(SGDSerialClientTrainer, SerialClientAlogrithmBase):
    def __init__(
        self,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        wandb_logger: WandbLogger=None,
        personal=False,
        args=None,
    ) -> None:
        SGDSerialClientTrainer.__init__(
            self, model, num_clients, cuda, device, logger, personal,
        )
        SerialClientAlogrithmBase.__init__(self)
        self.cache = []
        self.args = args
        self.wandb_logger = wandb_logger

        self.cur_global_model = deepcopy(self._model)

        self.cur_payload = None  

        self.is_prox = self.args.use_fedprox
        
        self.set_hooks() 

        self.on_init()

    def set_hooks(self):
        self.register_hooks(TestHook(test_interval=5), None, "LOWEST")
        self.register_hooks(EvaluateTrainHook(model=self.model, log_annotation="local", eval_interval=5), "local_eval", "LOWEST")
        self.register_hooks(EvaluateTrainHook(model=self.cur_global_model, log_annotation="global", eval_interval=5), "global_eval", "LOWEST")
        if self.args.grad_clip:
            self.register_hooks(ClientGradClipHook(clip_grad_norm=self.args.clip_grad_norm), None, "LOWEST")
        if type(self) == FedAPClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def set_global_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self.cur_global_model, parameters)

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self._model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args)
    
        # used for initialization for lr_scheduler
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.lr_scheduler=get_lr_scheduler(args=self.args,optimizer=self.optimizer,last_epoch=(self.round - 1) if hasattr(self,"round") else -1)
    
    def set_global_cid(self, local_id_list, rank):
        global_id_list = local_id_list + (rank - 1) * self.num_clients
        self.global_id_list = global_id_list.tolist()

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list, cur_round):
        self.id_list = id_list

        # get payloads
        self.cur_payload = payload
        self.round = cur_round
        model_parameters = payload[0]  

        self._LOGGER.info(f"Round {self.round} selected clients global id: {self.id_list}")

        self.set_global_model(model_parameters)

        self.on_local_process_start() # call self hook stage after get something
        
        # serial local training
        for cid in self.id_list:
            self.g_cid = cid
            self.l_cid = cid
            data_loader = self.dataset.get_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)

            self.on_client_serial_process_start()
            
            pack = self.train(model_parameters, data_loader)            
            self.cache.append(pack)

            self.on_client_serial_process_end()

        self.on_local_process_end()
        
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        if self.is_prox:
            frz_model = deepcopy(self.model)
        self.setup_optim(self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum)
        self.model.train()
        
        data_size = len(train_loader.dataset)
        data_size = torch.tensor(data_size)
        g_cid = torch.tensor(self.g_cid)

        self.on_client_training_start()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        loss_ = AverageMeter()
        for epoch in range(self.epochs):
            self.on_training_epoch_start()

            self.model.train()
            
            loss_.reset()

            for batch in train_loader:
                self.on_training_batch_start()
                
                imgs, labels, noisy_labels, guids = batch["img"], batch["label"], batch["noisy_label"], batch["guid"]
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, noisy_labels)

                if self.is_prox:
                    l2 = 0.0
                    mu = 0.1 # todo: automatic adjustment
                    for w0, w in zip(frz_model.parameters(), self.model.parameters()):
                        l2 += torch.sum(torch.pow(w - w0, 2))
                    loss += 0.5 * mu * l2
                
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss.backward()

                self.on_optimize_step_start()
                
                self.optimizer.step()

                self.on_optimize_step_end()
                
                loss_.update(loss.item())
                
                self.on_training_batch_end()

            self.on_training_epoch_end()
            
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} local train epoch [{epoch}/{self.epochs}], loss: {loss_.avg:.4f}"
            )

        self.on_client_training_end()

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} local training done."
        )

        local_result = [self.model_parameters, data_size, g_cid]
        return local_result