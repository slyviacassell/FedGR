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
    SSLWeightSchedulerHook,
    GlobalEMAHook,
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
        if self.args.anchor_model == "global_ema":
            self.register_hooks(GlobalEMAHook(), "shared_anchor", "LOWEST")
            self.register_hooks(TestHook(model_attr="anchor_model", log_annotation="global ema"), 'ema_test', "LOWEST")
        if self.args.local_ema:
            self.register_hooks(SerialClientLocalEMAHook(), "local_ema", "LOWEST")

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
            data_loader = self.dataset.get_semiws_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size, num_workers=self.args.num_workers)

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
            elif isinstance(hook, SampleMetricEvalClientHook):
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
            elif isinstance(hook, LocalOrchestra):
                self.global_centroids.to(self.device)
            elif isinstance(hook, EvaluateTrainHook):
                if hook.log_annotation == "global":
                    hook.model = self.cur_global_model
                elif hook.log_annotation == "local":
                    hook.model = self.model
            elif isinstance(hook, TestHook):
                if hook_name == "ema_test":
                    hook.model = self.anchor_model

    # === utils ===

    def set_hooks(self):
        self.register_hooks(SampleMetricEvalClientHook(), None, "LOWEST")

        if self.args.ckpt:
            self.register_hooks(FedNLLClientCheckPointHook(ckpt_interval=self.args.ckpt_interval), 'cli_ckpt', "LOWEST")

        self.register_hooks(SemiSupLoss(), "sl_loss", "LOWEST")
        self.register_hooks(SLWeightSchedulerHook(), "sl_weight_scheduler", "LOWEST")
        self.register_hooks(SSLWeightSchedulerHook(), "ssl_weight_scheduler", "LOWEST")
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
        if not self.args.use_local_mixup:
            imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
            clean_mask = noisy_labels == labels
            if self.cuda:
                imgs_w = imgs_w.to(self.device)
                imgs_s = imgs_s.to(self.device)
                noisy_labels = noisy_labels.to(self.device)

            outputs_s = self.model(imgs_s, return_dict=True, full_heads=True)
            with torch.no_grad():
                if self.args.anchor_model == "global_ema":
                    outputs_w = self.anchor_model(imgs_w, return_dict=True, full_heads=True)
                elif self.args.anchor_model == "global":
                    self.cur_global_model.eval()
                    outputs_w = self.cur_global_model(imgs_w, return_dict=True, full_heads=True)
            
            # targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["simplessl_head"]}
            targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["cls_head"]["cls_embedding"]}
            ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)
            sl_loss  = self.call_hook("loss", "sl_loss", outputs_s, targets, guids=guids, labels=labels, clean_mask=clean_mask)
        else:
            imgs_w, imgs_s, labels, noisy_labels, guids = batch["img_w"], batch["img_s"], batch["label"], batch["noisy_label"], batch["guid"]
            
            p_mask = torch.tensor([True if g in self.hooks_dict['sl_loss'].relabels[self.g_cid] else False for g in guids.numpy().tolist()]).to(self.device)
            c_probs = torch.tensor([self.hooks_dict['sl_loss'].sample_probs.get(g, 1) for g in guids.numpy().tolist()], device=self.device, dtype=torch.float32)
            noisy_mask = torch.tensor([True if g in self.overall_noisy_guids else False for g in guids.numpy()]).to(self.device)

            if self.cuda:
                imgs_w = imgs_w.to(self.device)
                imgs_s = imgs_s.to(self.device)
                noisy_labels = noisy_labels.to(self.device)

            alpha = self.args.mixup_alpha
            mixup_alpha = np.random.beta(alpha, alpha)
            mix_idx = torch.randperm(imgs_w.size(0))
            mixed_w = imgs_w * mixup_alpha + (1. - mixup_alpha) * imgs_w[mix_idx]
            mixed_s = imgs_s * mixup_alpha + (1. - mixup_alpha) * imgs_s[mix_idx]

            outputs_s = self.model(mixed_s, return_dict=True, full_heads=True)
            with torch.no_grad():
                if self.args.anchor_model == "global_ema":
                    outputs_w = self.anchor_model(mixed_w, return_dict=True, full_heads=True)
                elif self.args.anchor_model == "global":
                    outputs_w = self.cur_global_model(mixed_w, return_dict=True, full_heads=True)

            if self.round < self.args.warmup_round + self.args.sniffing_round:
                # targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["simplessl_head"]}
                targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["cls_head"]["cls_embedding"]}
                ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)
                sl_loss1  = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], noisy_labels)
                sl_loss2  = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], noisy_labels[mix_idx])
                sl_loss = mixup_alpha * sl_loss1 + (1 - mixup_alpha) * sl_loss2
            else:
                n_rate = self.hooks_dict["sl_loss"].est_cid_noise[self.g_cid]
                # targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["simplessl_head"]}
                targets = {"cls_head": noisy_labels, "simplessl_head": outputs_w["cls_head"]["cls_embedding"]}
                ssl_loss, ssl2sl_reg = self.call_hook("loss", "ssl_loss", outputs_s, targets)

                sharpen_temp = 0.5
                if n_rate > self.args.upper_rate_threshold:
                    pseudo_labels = [self.hooks_dict["sl_loss"].relabels[self.g_cid].get(g, -100) for i, g in enumerate(guids.numpy().tolist())]
                    pseudo_labels = torch.tensor(pseudo_labels).to(self.device)
                    
                    soft_pse_targets = np.array([self.hooks_dict["sl_loss"].soft_pse_labels[self.g_cid].get(g, np.zeros(CLASS_NUM[self.args.dataset])) for i, g in enumerate(guids.numpy().tolist())])
                    soft_pse_targets = torch.tensor(soft_pse_targets, device=self.device)
                    soft_loss = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs_s["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)

                    if (pseudo_labels != -100).sum() > 0:
                        sl_loss_1 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], pseudo_labels, reduction='none')
                        sl_loss_2 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], pseudo_labels[mix_idx], reduction='none')
                        sl_loss = mixup_alpha * sl_loss_1 + (1 - mixup_alpha) * sl_loss_2
                        sl_loss = sl_loss.mean()
                    else:
                        sl_loss = torch.tensor(0., device=self.device)
                    
                    sl_loss += soft_loss.mean()
                else:
                    pseudo_labels = [self.hooks_dict["sl_loss"].relabels[self.g_cid].get(g, -100) for i, g in enumerate(guids.numpy().tolist())]
                    pseudo_labels = torch.tensor(pseudo_labels).to(self.device)

                    soft_pse_targets = np.array([self.hooks_dict["sl_loss"].soft_pse_labels[self.g_cid].get(g, np.zeros(CLASS_NUM[self.args.dataset])) for i, g in enumerate(guids.numpy().tolist())])
                    soft_pse_targets = torch.tensor(soft_pse_targets, device=self.device)
                    soft_loss_1 = -torch.sum(torch.softmax(soft_pse_targets/sharpen_temp, dim=-1) * torch.log_softmax(outputs_s["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
                    soft_loss_2 = -torch.sum(torch.softmax(soft_pse_targets[mix_idx]/sharpen_temp, dim=-1) * torch.log_softmax(outputs_s["cls_head"]["cls_logits"]/sharpen_temp+1e-10, dim=1), dim=1)
                    soft_loss = mixup_alpha * soft_loss_1 + (1 - mixup_alpha) * soft_loss_2
                    soft_loss = soft_loss.mean()

                    noisy_loss_1 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], noisy_labels, ignore_index=-100, reduction="none")
                    pse_loss_1 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], pseudo_labels, ignore_index=-100, reduction="none")
                    sl_loss_1 = noisy_loss_1 * c_probs + pse_loss_1 * (1. - c_probs)
                    sl_loss_1 = sl_loss_1.mean()

                    noisy_loss_2 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], noisy_labels[mix_idx], ignore_index=-100, reduction="none")
                    pse_loss_2 = TF.cross_entropy(outputs_s["cls_head"]["cls_logits"], pseudo_labels[mix_idx], ignore_index=-100, reduction="none")
                    sl_loss_2 = noisy_loss_2 * c_probs[mix_idx] + pse_loss_2 * (1. - c_probs[mix_idx])
                    sl_loss_2 = sl_loss_2.mean()

                    sl_loss = mixup_alpha * sl_loss_1 + (1 - mixup_alpha) * sl_loss_2 + soft_loss
            
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
            ssl_loss, sl_loss, ssl2sl_reg = self.orchesta_step(batch, *args, **kwargs)
        elif self.args.ssl_method == "simsiam":
            ssl_loss, sl_loss, ssl2sl_reg = self.simsiam_step(batch, *args, **kwargs)
        elif self.args.ssl_method == "byol":
            ssl_loss, sl_loss, ssl2sl_reg = self.byol_step(batch, *args, **kwargs)
        elif self.args.ssl_method == "simplessl":
            ssl_loss, sl_loss, ssl2sl_reg = self.simplessl_step(batch, *args, **kwargs)
        elif self.args.ssl_method == "fedprox_like":
            ssl_loss, sl_loss, ssl2sl_reg = self.fedprox_like_step(batch, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized ssl method: {self.args.ssl_method}")
        
        sl_weight = self.call_hook("step", "sl_weight_scheduler", sl_weight=self.args.sl_weight)
        ssl_weight = self.call_hook("step", "ssl_weight_scheduler", ssl_weight=self.args.ssl_weight)
        ssl2sl_reg_weight = self.args.ssl2sl_reg_weight

        tot_loss = ssl_loss * ssl_weight + sl_loss * sl_weight + ssl2sl_reg * ssl2sl_reg_weight
        return tot_loss, ssl_loss * ssl_weight, sl_loss * sl_weight, ssl2sl_reg * ssl2sl_reg_weight
    # === Forward Step ===

    def train_epoch(self, train_loader, mu=0.0, *args, **kwargs):
        self.naive_epoch(train_loader, mu=mu, *args, **kwargs)

    def naive_epoch(self, train_loader, mu=0.0, *args, **kwargs):
        tot_loss_meter = kwargs.get("tot_loss_meter", AverageMeter())
        ssl_loss_meter = kwargs.get("ssl_loss_meter", AverageMeter())
        sl_loss_meter = kwargs.get("sl_loss_meter", AverageMeter())
        reg_loss_meter = kwargs.get("reg_loss_meter", AverageMeter())

        # tmp
        self.local_ema_meter = AverageMeter()

        if self.is_prox:
            # frz_model = deepcopy(self.model)
            self.cur_global_model.eval()
            frz_model = self.cur_global_model

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

            self.train_epoch(
                train_loader, 
                mu=mu, 
                tot_loss_meter=tot_loss_meter, 
                ssl_loss_meter=ssl_loss_meter, 
                sl_loss_meter=sl_loss_meter, 
                reg_loss_meter=reg_loss_meter,
            )
            
            self.on_training_epoch_end()
            
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} e [{epoch}/{self.epochs}], "
                f"tot_loss:{tot_loss_meter.avg:.4f},"
                f"ssl_loss:{ssl_loss_meter.avg:.4f},"
                f"sl_loss:{sl_loss_meter.avg:.4f},"
                f"reg_loss:{reg_loss_meter.avg:.4f},"
                # tmp
                f"ema_acc: {self.local_ema_meter.avg*100:.2f}%"
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
    