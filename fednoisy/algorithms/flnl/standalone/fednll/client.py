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
from fednoisy.algorithms.flnl.hooks import (
    SampleMetricEvalClientHook,
    LabelNoiseMaskOutLoss,
    LabelNoiseOrcaleMaskOutLoss,
    LabelNoiseTruncationLoss,
    LabelNoiseWeight,
    ClientLabelDistriEMA,
)
from fednoisy.algorithms.flnl.standalone.fednll.hooks import (
    LocalOrchestra,
    FedNLLClientCheckPointHook,
    OrchestraLoss,
    SemiSupLoss,
    SimSiamLoss,
    BYOLLoss,
    SimpleSSLLoss,
    SLWeightSchedulerHook,
    SharedAnchorHook,
)
from torch.profiler import profile, record_function, ProfilerActivity


class FedAPNLLClientTrainer(FedAPClientTrainer):
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

        super(FedAPNLLClientTrainer, self).__init__(
            model, num_clients, cuda, device, logger, wandb_logger, personal, args
        )

    # === SSL Hooks ===
    def register_fedprox_like_hooks(self):
        self.register_hooks(SimpleSSLLoss(), "ssl_loss", "LOWEST")
        self.register_hooks(SharedAnchorHook(), "shared_anchor", "LOWEST")

    def register_simple_ssl_hooks(self):
        # assert self.args.local_ema, "Local EMA should be enabled for Simple SSL"
        self.register_hooks(SerialClientLocalEMAHook(), "local_ema", "LOWEST")
        self.register_hooks(SimpleSSLLoss(), "ssl_loss", "LOWEST")

    def register_orchestra_hooks(self):
        # assert self.args.local_ema, "Local EMA should be enabled for Orchestra"
        self.register_hooks(SerialClientLocalEMAHook(), "local_ema", "LOWEST")
        self.register_hooks(LocalOrchestra(), "local_orchestra", "LOWEST")
        self.register_hooks(OrchestraLoss(), "ssl_loss", "LOWEST")

    def register_simsiam_hooks(self):
        self.register_hooks(SimSiamLoss(), "ssl_loss", "LOWEST")

    def register_byol_hooks(self):
        # assert self.args.local_ema, "Local EMA should be enabled for BYOL"
        self.register_hooks(SerialClientLocalEMAHook(), "local_ema", "LOWEST")
        self.register_hooks(BYOLLoss(), "ssl_loss", "LOWEST")

    def register_ssl_hooks(self):
        if self.args.ssl_method == "orchestra":
            self.register_orchestra_hooks()
        elif self.args.ssl_method == "simsiam":
            self.register_simsiam_hooks()
        elif self.args.ssl_method == "byol":
            self.register_byol_hooks()
        elif self.args.ssl_method == "simplessl":
            self.register_simple_ssl_hooks()
        elif self.args.ssl_method == "fedprox_like":
            self.register_fedprox_like_hooks()
        else:
            raise ValueError(f"Unrecognized ssl method: {self.args.ssl_method}")
    # === SSL Hooks ===
    
    # === utils ===
    def get_dataloader(self):
        if self.args.ssl_method == "orchestra" or self.args.ssl_method == "simplessl":
            data_loader = self.dataset.get_semiws_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)
        elif self.args.ssl_method == "simsiam" or self.args.ssl_method == "byol":
            data_loader = self.dataset.get_dividemix_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)
        elif self.args.ssl_method == "fedprox_like":
            data_loader = self.dataset.get_semiws_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)
            # todo
            # if self.round < self.args.warmup_round + self.args.sniffing_round:
            #     data_loader = self.dataset.get_semiws_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)
            # else:
            #     data_loader = self.dataset.get_dividemix_dataloader(
            #         cid=self.g_cid, 
            #         train=True,
            #         batch_size=self.batch_size,
                    
            #     )
            
        return data_loader
    
    def packing_local_results(self, local_results: List):
        if self.args.ssl_method == "orchestra":
            local_results = local_results + self.local_centroids
        return local_results
    
    def __getstate__(self):
        ckpt_hooks = {}
        for hook_name, hook in self.hooks_dict.items():
            if isinstance(hook, SerialClientLocalEMAHook):
                for m in self.local_ema_models:
                    m.ema_model.to("cpu")
                ckpt_hooks[hook_name] = hook
            elif isinstance(hook, LocalOrchestra):
                self.global_centroids.to("cpu")
                ckpt_hooks[hook_name] = hook

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
                    m.ema_model.to(self.device)
            elif isinstance(hook, LocalOrchestra):
                self.global_centroids.to(self.device)
    # === utils ===

    def set_hooks(self):
        self.register_hooks(SampleMetricEvalClientHook(), None, "LOWEST")

        if self.args.ckpt:
            self.register_hooks(FedNLLClientCheckPointHook(ckpt_interval=self.args.ckpt_interval), 'cli_ckpt', "LOWEST")

        self.register_hooks(SemiSupLoss(), "sl_loss", "LOWEST")
        self.register_hooks(SLWeightSchedulerHook(), "sl_weight_scheduler", "LOWEST")
        self.register_ssl_hooks()

        super(FedAPNLLClientTrainer, self).set_hooks()

        if type(self) ==  FedAPNLLClientTrainer:
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
    def fedprox_like_step(self, batch, *args, **kwargs):
        imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            imgs_w = imgs_w.to(self.device)
            imgs_s = imgs_s.to(self.device)
            noisy_labels = noisy_labels.to(self.device)

        outputs_s = self.model(imgs_s, return_dict=True, full_heads=True)
        with torch.no_grad():
            self.cur_global_model.eval()
            # outputs_w = self.cur_global_model(imgs_w, return_dict=True, full_heads=True)
            outputs_w = self.anchor_model(imgs_w, return_dict=True, full_heads=True)
        targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["simplessl_head"]}
        ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)
        sl_loss  = self.call_hook("loss", "sl_loss", outputs_s, targets, guids=guids, labels=labels, clean_mask=clean_mask)

        return ssl_loss, sl_loss, ssl2sl_reg

    def simplessl_step(self, batch, *args, **kwargs):
        imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            imgs_w = imgs_w.to(self.device)
            imgs_s = imgs_s.to(self.device)
            noisy_labels = noisy_labels.to(self.device)

        outputs_s = self.model(imgs_s, return_dict=True, full_heads=True)
        outputs_w = self.call_hook("ema_outputs", "local_ema", inputs=imgs_w, return_dict=True, full_heads=True)
        targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["simplessl_head"]}

        ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)
        sl_loss  = self.call_hook("loss", "sl_loss", outputs_s, targets)

        return ssl_loss, sl_loss, ssl2sl_reg
    
    def orchesta_step(self, batch, *args, **kwargs):
        imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            imgs_w = imgs_w.to(self.device)
            imgs_s = imgs_s.to(self.device)
            noisy_labels = noisy_labels.to(self.device)

        outputs_s = self.model(imgs_s, return_dict=True, full_heads=True)
        outputs_s["orchestra_head"] = self.global_centroids(TF.normalize(outputs_s["orchestra_head"], dim=1))
        outputs_w = self.call_hook("ema_outputs", "local_ema", inputs=imgs_w, return_dict=True, full_heads=True)
        q = self.call_hook("get_assignment", "local_orchestra", outputs_w=outputs_w, outputs_s=outputs_s, guids=guids, labels=labels, clean_mask=clean_mask)
        targets = {"cls_head": noisy_labels, "orchestra_head": q}

        ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)
        sl_loss  = self.call_hook("loss", "sl_loss", outputs_s, targets)

        return ssl_loss, sl_loss, ssl2sl_reg

    def simsiam_step(self, batch, *args, **kwargs):
        x1, x2, labels, noisy_labels, guids = batch["img_0"], batch["img_1"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            noisy_labels = noisy_labels.to(self.device)
        
        x = torch.cat([x1, x2], dim=0)
        outputs = self.model(x, return_dict=True, full_heads=True)
        ssl_outputs = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0),
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0),
            },
            "projector_predictor": {
                "z": outputs["projector_predictor"]["z"].chunk(2, dim=0),
                "p": outputs["projector_predictor"]["p"].chunk(2, dim=0),
            },
        }
        sl_outputs1 = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0)[0],
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0)[0],
            },
        }
        sl_outputs2 = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0)[1],
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0)[1],
            },
        }
        sl_targets = {"cls_head": noisy_labels}
        
        ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", ssl_outputs)
        sl_loss1  = self.call_hook("loss", "sl_loss", sl_outputs1, sl_targets)
        sl_loss2  = self.call_hook("loss", "sl_loss", sl_outputs2, sl_targets)
        sl_loss = (sl_loss1 + sl_loss2) / 2

        return ssl_loss, sl_loss, ssl2sl_reg
    
    def byol_step(self, batch, *args, **kwargs):
        x1, x2, labels, noisy_labels, guids = batch["img_0"], batch["img_1"], batch["label"], batch["noisy_label"], batch["guid"]
        clean_mask = noisy_labels == labels
        if self.cuda:
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            noisy_labels = noisy_labels.to(self.device)

        x = torch.cat([x1, x2], dim=0)
        outputs = self.model(x, return_dict=True, full_heads=True)
        ema_outputs = self.call_hook("ema_outputs", "local_ema", inputs=x, return_dict=True, full_heads=True)
        ssl_outputs = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0),
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0),
            },
            "projector_predictor": {
                "z": outputs["projector_predictor"]["z"].chunk(2, dim=0),
                "p": outputs["projector_predictor"]["p"].chunk(2, dim=0),
            },
        }
        ssl_targets = {
            "projector_predictor": {
                "z": ema_outputs["projector_predictor"]["z"].chunk(2, dim=0),
                "p": ema_outputs["projector_predictor"]["p"].chunk(2, dim=0),
            },
        }
        sl_outputs1 = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0)[0],
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0)[0],
            },
        }
        sl_outputs2 = {
            "cls_head": {
                "cls_logits": outputs["cls_head"]["cls_logits"].chunk(2, dim=0)[1],
                "cls_embedding": outputs["cls_head"]["cls_embedding"].chunk(2, dim=0)[1],
            },
        }
        sl_targets = {
            "cls_head": noisy_labels, 
        }

        ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", ssl_outputs, ssl_targets)
        sl_loss1  = self.call_hook("loss", "sl_loss", sl_outputs1, sl_targets)
        sl_loss2  = self.call_hook("loss", "sl_loss", sl_outputs2, sl_targets)
        sl_loss = (sl_loss1 + sl_loss2) / 2
        return ssl_loss, sl_loss, ssl2sl_reg
    
    def forward_step(self, batch, *args, **kwargs):
        if self.args.ssl_method == "orchestra":
            ssl_loss, sl_loss, ssl2sl_reg = self.orchesta_step(batch)
        elif self.args.ssl_method == "simsiam":
            ssl_loss, sl_loss, ssl2sl_reg = self.simsiam_step(batch)
        elif self.args.ssl_method == "byol":
            ssl_loss, sl_loss, ssl2sl_reg = self.byol_step(batch)
        elif self.args.ssl_method == "simplessl":
            ssl_loss, sl_loss, ssl2sl_reg = self.simplessl_step(batch)
        elif self.args.ssl_method == "fedprox_like":
            ssl_loss, sl_loss, ssl2sl_reg = self.fedprox_like_step(batch)
        else:
            raise ValueError(f"Unrecognized ssl method: {self.args.ssl_method}")
        
        sl_weight = self.call_hook("step", "sl_weight_scheduler", sl_weight=self.args.sl_weight)

        tot_loss = ssl_loss * self.args.ssl_weight + sl_loss * sl_weight + ssl2sl_reg * self.args.ssl2sl_reg_weight
        return tot_loss, ssl_loss, sl_loss, ssl2sl_reg
    # === Forward Step ===
        
    def train(self, model_parameters, train_loader, mu=0.0):
        self.set_model(model_parameters)
        if self.is_prox:
            # frz_model = deepcopy(self.model)
            self.cur_global_model.eval()
            frz_model = self.cur_global_model
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

        tot_loss_meter = AverageMeter()
        ssl_loss_meter = AverageMeter()
        sl_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()

        for epoch in range(self.epochs):
            self.on_training_epoch_start()

            self.model.train()
            
            tot_loss_meter.reset()
            ssl_loss_meter.reset()
            sl_loss_meter.reset()
            reg_loss_meter.reset()

            for batch in train_loader:
                self.on_training_batch_start()

                loss, ssl_loss, sl_loss, ssl2sl_reg = self.forward_step(batch)

                if self.is_prox:
                    self.call_hook("update", "fedprox_loss_meter", l_cid=self.l_cid, batch_loss=loss.item())
                    l2 = 0.0
                    for w0, w in zip(frz_model.parameters(), self.model.parameters()):
                        l2 += torch.sum(torch.pow(w - w0, 2))
                    loss += 0.5 * mu * l2

                bsz = len(batch)
                tot_loss_meter.update(loss.item(), bsz)
                ssl_loss_meter.update(ssl_loss.item(), bsz)
                sl_loss_meter.update(sl_loss.item(), bsz)
                reg_loss_meter.update(ssl2sl_reg.item(), bsz)
                
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss.backward()

                self.on_optimize_step_start()
                
                self.optimizer.step()

                self.on_optimize_step_end()
                
                tot_loss_meter.update(loss.item())
                
                self.on_training_batch_end()

            self.on_training_epoch_end()
            
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} epoch [{epoch}/{self.epochs}], "
                f"tot_loss:{tot_loss_meter.avg:.4f}, "
                f"ssl_loss:{ssl_loss_meter.avg:.4f}, "
                f"sl_loss:{sl_loss_meter.avg:.4f}, "
                f"reg_loss:{reg_loss_meter.avg:.4f}"
            )

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
    