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

from fednoisy.algorithms.flnl.scale.fedap import FedAPClientTrainer
from fednoisy.algorithms.flnl.hooks import (
    SampleMetricEvalClientHook,
    LabelNoiseMaskOutLossHook,
    FedProxLocalLossMeterHook,
)

class FedAPCSClientTrainer(FedAPClientTrainer):
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
        # custom attributes

        super(FedAPCSClientTrainer, self).__init__(
            model, num_clients, cuda, device, logger, wandb_logger, personal, args
        )

    def set_hooks(self):
        self.register_hooks(SampleMetricEvalClientHook(), None, "LOWEST")

        if self.args.mask_out_loss:
            self.register_hooks(LabelNoiseMaskOutLossHook(), "mask_out_loss","LOWEST")

        if self.args.use_fedprox:
            self.register_hooks(FedProxLocalLossMeterHook(self.args), "fedprox_loss_meter", "LOWEST")

        super(FedAPCSClientTrainer, self).set_hooks()

        if type(self) ==  FedAPCSClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )

    def local_process(self, payload, id_list, cur_round, rank):
        self.local_id_list = id_list.tolist()
        self.rank = rank.item()

        # get payloads
        self.cur_payload = payload
        self.round = cur_round.item()
        model_parameters = payload[0]  
        mu = payload[1]

        self.set_global_model(model_parameters)

        self.set_global_cid(id_list, rank)

        self._LOGGER.info(f"Round {self.round} selected clients global id: {self.global_id_list}, selected clients local id: {self.local_id_list}")

        self.on_local_process_start() # call self hook stage after get something
        
        # serial local training
        for l_cid, g_cid in zip(self.local_id_list, self.global_id_list):
            self.l_cid = l_cid
            self.g_cid = g_cid
            data_loader = self.dataset.get_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)

            self.on_client_serial_process_start()
            
            pack = self.train(model_parameters, data_loader, mu)            
            self.cache.append(pack)

            self.on_client_serial_process_end()

        self.on_local_process_end()
        
    def train(self, model_parameters, train_loader, mu=0.0):
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
                if self.hooks_dict.get("mask_out_loss", None) is not None:
                    loss = self.call_hook("loss", "mask_out_loss", outputs, noisy_labels, guids=guids)
                else:
                    loss = self.criterion(outputs, noisy_labels)

                if self.is_prox:
                    self.call_hook("update", "fedprox_loss_meter", self.l_cid, loss.item())
                    l2 = 0.0
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

        if self.args.use_fedprox:
            local_loss = [self.call_hook("get_loss_avg", "fedprox_loss_meter", self.l_cid)]
        else:
            local_loss = 0.0
        local_loss = torch.tensor(local_loss)

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} local training done."
        )

        local_result = [self.model_parameters, data_size, local_loss, g_cid] + self.cs_metrics
        return local_result