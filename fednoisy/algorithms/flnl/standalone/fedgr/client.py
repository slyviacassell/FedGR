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

from fednoisy.algorithms.flnl.standalone.fedap import FedAPClientTrainer
from fednoisy.algorithms.flnl.standalone.fednll.hooks import (
    FedNLLClientCheckPointHook,
    SimpleSSLLoss,
)
from fednoisy.algorithms.flnl.standalone.fedgr.hooks import (
    SLWeightSchedulerHook,
    SSLWeightSchedulerHook,
    EMADistillWeightSchedulerHook,
)
from fednoisy.algorithms.flnl.standalone.fedgr.hooks import (
    SniffAndRefineClientHook,
    SniffAndRefineServerHook,
    EMADistillClientHook,
    RepresentaionRegHook,
)
from torch.profiler import profile, record_function, ProfilerActivity


class FedGRClientTrainer(FedAPClientTrainer):
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

        super(FedGRClientTrainer, self).__init__(
            model, num_clients, cuda, device, logger, wandb_logger, personal, args
        )

    # === SSL Hooks ===
    def register_fedgr_hooks(self):
        self.register_hooks(SniffAndRefineClientHook(), "sl_loss", "LOWEST")
        self.register_hooks(RepresentaionRegHook(), "ssl_loss", "LOWEST")
        self.register_hooks(EMADistillClientHook(), "ema_distill_loss", "LOW")
        if self.args.local_ema:
            self.register_hooks(SerialClientLocalEMAHook(), "local_ema", "LOWEST")
        self.register_hooks(SLWeightSchedulerHook(), "sr_weight_scheduler", "LOWEST")
        self.register_hooks(SSLWeightSchedulerHook(), "reg_weight_scheduler", "LOWEST")
        self.register_hooks(EMADistillWeightSchedulerHook(), "ema_distill_weight_scheduler", "LOWEST")

    def register_ssl_hooks(self):
        if self.args.ssl_method == "fedprox_like":
            self.register_fedgr_hooks()
        else:
            raise ValueError(f"Unrecognized ssl method: {self.args.ssl_method}")
    # === SSL Hooks ===
    
    # === utils ===
    def get_dataloader(self):
        if self.args.ssl_method == "fedprox_like":
            data_loader = self.dataset.get_semiws_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size, num_workers=self.args.num_workers)

        return data_loader
    
    def packing_local_results(self, local_results: List):
        return local_results
    
    def __getstate__(self):
        ckpt_hooks = {}
        for hook_name, hook in self.hooks_dict.items():
            if isinstance(hook, SerialClientLocalEMAHook):
                for m in self.local_ema_models:
                    m.ema_model.to("cpu")
                ckpt_hooks[hook_name] = hook
            elif isinstance(hook, SniffAndRefineClientHook):
                ckpt_hooks[hook_name] = hook
        self._LOGGER.info(f"Client checkpoint hooks: {ckpt_hooks.keys()}")

        self.model.to("cpu")

        state = {k: v for k, v in self.__dict__.items() if k not in self._blacklist}
        state["ckpt_hooks"] = ckpt_hooks
        return state
    
    def load_state(self, state):
        vaild_state = {k: v for k, v in state.items() if k not in ["_hooks", "hooks_dict"]}
        for k,v in state["hooks_dict"].items():
            self.register_hooks(v, k, v.priority)
            self._LOGGER.info(f"Load hook: {k}")
        self.__dict__.update(vaild_state)
        self._LOGGER.info(f"Load state: {vaild_state.keys()}")

        self.model.to(self.device)
        for hook_name, hook in self.hooks_dict.items():
            if isinstance(hook, SerialClientLocalEMAHook):
                for m in self.local_ema_models:
                    m.online_model = self.model
                    # m.ema_model.to(self.device)
            elif isinstance(hook, EvaluateTrainHook):
                if hook.log_annotation == "global":
                    hook.model = self.cur_global_model
                elif hook.log_annotation == "local":
                    hook.model = self.model
    # === utils ===

    def set_hooks(self):
        if self.args.ckpt:
            self.register_hooks(FedNLLClientCheckPointHook(ckpt_interval=self.args.ckpt_interval), 'cli_ckpt', "LOWEST")

        self.register_ssl_hooks()

        super(FedGRClientTrainer, self).set_hooks()

        if type(self) ==  FedGRClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )
        self._LOGGER.info(
            f"ssl_method: {self.args.ssl_method}"
        )

    def local_process(self, payload, id_list, cur_round):
        self.id_list = id_list

        # get payloads
        self.cur_payload = payload
        self.round = cur_round
        model_parameters = payload[0]  
        mu = payload[1]

        self._LOGGER.info(f"Round {self.round} selected clients global id: {self.id_list}")

        self.set_global_model(model_parameters)

        self.on_local_process_start() # call self hook stage after get something
        
        # serial local training
        for cid in self.id_list:
            self.g_cid = cid
            self.l_cid = cid
            data_loader = self.get_dataloader()

            self.on_client_serial_process_start()
            
            pack = self.train(model_parameters, data_loader, mu)            
            self.cache.append(pack)

            self.on_client_serial_process_end()

        self.on_local_process_end()

    # === Forward Step ===
    def fedgr_step(self, batch, *args, **kwargs):
        imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            imgs_w = imgs_w.to(self.device, non_blocking=True)
            imgs_s = imgs_s.to(self.device, non_blocking=True)
            noisy_labels = noisy_labels.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

        outputs_s = self.model(imgs_s, return_dict=True, full_heads=True)
        with torch.no_grad():
            outputs_w = self.cur_global_model(imgs_w, return_dict=True, full_heads=True)
            if self.args.use_online_ema:
                ema_outputs_w = self.local_ema_models[self.l_cid](imgs_w, return_dict=True, full_heads=True) # online ema
            else:
                ema_outputs_w = torch.stack([self.cid_soft_targets[self.l_cid][g.item()] for g in guids], dim=0).to(self.device, non_blocking=True) # global revised ema
        
        if self.args.use_online_ema:
            targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["cls_head"]["cls_embedding"], "ema_distill_targets": ema_outputs_w["cls_head"]["cls_logits"]} # online ema
        else:
            targets = {"clean_labels":labels, "cls_head": noisy_labels, "simplessl_head": outputs_w["cls_head"]["cls_embedding"], "ema_distill_targets": ema_outputs_w} # global revised ema
        reg_loss = self.call_hook("loss", "ssl_loss", outputs_s, targets)
        ema_distill_loss = self.call_hook("loss", "ema_distill_loss", outputs_s, targets)
        sr_loss = self.call_hook("loss", "sl_loss", outputs_s, targets, guids=guids, labels=labels, clean_mask=clean_mask)

        return reg_loss, sr_loss, ema_distill_loss
    
    def forward_step(self, batch, *args, **kwargs):
        if self.args.ssl_method == "fedprox_like":
            reg_loss, sr_loss, ema_distill_loss = self.fedgr_step(batch, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized ssl method: {self.args.ssl_method}")
        
        sr_weight = self.call_hook("step", "sr_weight_scheduler", sl_weight=self.args.sl_weight)
        reg_weight = self.call_hook("step", "reg_weight_scheduler", ssl_weight=self.args.ssl_weight)
        ema_distill_weight = self.call_hook("step", "ema_distill_weight_scheduler", weight=self.args.ema_weight)

        tot_loss = reg_loss * reg_weight + sr_loss * sr_weight + ema_distill_loss * ema_distill_weight
        return tot_loss, reg_loss * reg_weight, sr_loss * sr_weight, ema_distill_loss * ema_distill_weight
    # === Forward Step ===

    def train_epoch(self, train_loader, mu=0.0, *args, **kwargs):
        self.naive_epoch(train_loader, mu=mu, *args, **kwargs)

    def naive_epoch(self, train_loader, mu=0.0, *args, **kwargs):
        tot_loss_meter = kwargs.get("tot_loss_meter", AverageMeter())
        reg_loss_meter = kwargs.get("reg_loss_meter", AverageMeter())
        sr_loss_meter = kwargs.get("sr_loss_meter", AverageMeter())
        ema_distill_meter = kwargs.get("ema_distill_meter", AverageMeter())

        # tmp
        self.local_ema_meter = AverageMeter()

        if self.is_prox:
            # frz_model = deepcopy(self.model)
            self.cur_global_model.eval()
            frz_model = self.cur_global_model

        for batch in train_loader:
            self.on_training_batch_start()

            loss, reg_loss, sr_loss, ema_distill_loss = self.forward_step(batch)

            if self.is_prox:
                self.call_hook("update", "fedprox_loss_meter", l_cid=self.l_cid, batch_loss=loss.item())
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self.model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))
                loss += 0.5 * mu * l2

            bsz = len(batch)
            tot_loss_meter.update(loss.item(), bsz)
            reg_loss_meter.update(reg_loss.item(), bsz)
            sr_loss_meter.update(sr_loss.item(), bsz)
            ema_distill_meter.update(ema_distill_loss.item(), bsz)
            
            self.optimizer.zero_grad()
            self.model.zero_grad()
            loss.backward()

            self.on_optimize_step_start()
            
            self.optimizer.step()

            self.on_optimize_step_end()
            
            tot_loss_meter.update(loss.item())
            
            self.on_training_batch_end()
        
    def train(self, model_parameters, train_loader, mu=0.0):
        self.set_model(model_parameters)
        self.setup_optim(self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum)
        self.model.train()
        
        data_size = len(train_loader.dataset)
        if self.hooks_dict.get("weight_adjustment", None) is not None:
            weight = self.call_hook("adjust_weight", "weight_adjustment", data_size)
        else:
            weight = data_size
        weight = torch.tensor(weight)
        g_cid = torch.tensor(self.g_cid)

        self.on_client_training_start()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.optimizer.param_groups[0]["lr"]
        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} lr: {self.optimizer.param_groups[0]['lr']}"
        )

        tot_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        sr_loss_meter = AverageMeter()
        ema_distill_meter = AverageMeter()

        for epoch in range(self.epochs):
            self.on_training_epoch_start()

            self.model.train()
            
            tot_loss_meter.reset()
            reg_loss_meter.reset()
            sr_loss_meter.reset()
            ema_distill_meter.reset()

            self.train_epoch(
                train_loader, 
                mu=mu, 
                tot_loss_meter=tot_loss_meter, 
                reg_loss_meter=reg_loss_meter, 
                sr_loss_meter=sr_loss_meter, 
                ema_distill_meter=ema_distill_meter,
            )
                        
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} e [{epoch}/{self.epochs}], "
                f"tot_loss:{tot_loss_meter.avg:.4f},"
                f"sr_loss:{sr_loss_meter.avg:.4f},"
                f"reg_loss:{reg_loss_meter.avg:.4f},"
                f"ema_dis_loss:{ema_distill_meter.avg:.4f},"
                # tmp
                f"ema_acc: {self.local_ema_meter.avg*100:.2f}%"
            )

            self.on_training_epoch_end()

        self.on_client_training_end()

        if self.is_prox:
            local_loss = self.call_hook("get_loss_avg", "fedprox_loss_meter", l_cid=self.l_cid)
        else:
            local_loss = 0.

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} local training done."
        )

        local_result = [self.model_parameters, weight, local_loss, g_cid] + self.cs_metrics 
        local_result = self.packing_local_results(local_result)
        return local_result
    