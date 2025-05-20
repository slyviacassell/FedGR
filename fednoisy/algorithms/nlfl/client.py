from ast import mod
from cgi import test
from re import L

from copy import deepcopy
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
from sklearn.mixture import GaussianMixture

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


class NLLFedAvgClientTrainer(SGDSerialClientTrainer, SerialClientAlogrithmBase):
    def __init__(
        self,
        backbone,
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

        if isinstance(backbone, nn.Identity):
            self.none_backbone = True
        else:
            self.none_backbone = False

        if cuda:
            self.backbone = backbone.cuda(self.device)
        else:
            self.backbone = backbone.cpu()

        self.set_hooks() 

        self.on_init()

    def set_hooks(self):
        if type(self) == NLLFedAvgClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )
    
    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def init_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.setup_optim(epochs, batch_size, lr, weight_decay, momentum)
        if not self.none_backbone:
            self.setup_backbone_optim(lr, weight_decay, momentum)

    def setup_backbone_optim(self, lr, weight_decay, momentum):
        self.backbone_optimizer = torch.optim.SGD(
            self.backbone.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )

        # used for initialization for lr_scheduler
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        
        self.backbone_lr_scheduler = get_lr_scheduler(
            args=self.args,
            optimizer=self.backbone_optimizer,
            last_epoch=(self.round - 1) if hasattr(self,"round") else -1
        )

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr, weight_decay=weight_decay, momentum=momentum
        )
        self.criterion = get_robust_loss(CLASS_NUM[self.args.dataset], self.args)
    
        # used for initialization for lr_scheduler
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.lr_scheduler = get_lr_scheduler(
            args=self.args,
            optimizer=self.optimizer,
            last_epoch=(self.round - 1) if hasattr(self,"round") else -1
        )

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package
    
    def local_process(self, payload, id_list, cur_round):
        model_parameters = payload[0]  

        self.round = cur_round

        self.on_local_process_start() # call self hook stage after get something
        
        # serial local training
        for cid in id_list:
            self.g_cid = cid

            data_loader = self.dataset.get_dataloader(cid=cid, train=True, batch_size=self.batch_size)

            self.on_client_serial_process_start()
            
            pack = self.train(model_parameters, data_loader)      
            self.noisy_eval(payload)  
            self.cache.append(pack)

            self.on_client_serial_process_end()

        self.on_local_process_end()

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum)

        model = nn.Sequential(self.backbone, self.model)
        model = model.train()

        data_size = len(train_loader.dataset)

        self.on_client_training_start()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if not self.none_backbone:
            if self.backbone_lr_scheduler is not None:
                self.backbone_lr_scheduler.step()

        loss_ = AverageMeter()
        for epoch in range(self.epochs):
            self.on_training_epoch_start()

            model.train()
            
            loss_.reset()

            for batch in train_loader:
                self.on_training_batch_start()
                
                imgs, labels, noisy_labels, guids = batch["img"], batch["label"], batch["noisy_label"], batch["guid"]
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = model(imgs)
                loss = self.criterion(outputs, noisy_labels)
                
                self.optimizer.zero_grad()
                if not self.none_backbone:
                    self.backbone_optimizer.zero_grad()
                model.zero_grad()

                loss.backward()

                self.on_optimize_step_start()
                
                self.optimizer.step()
                if not self.none_backbone:
                    self.backbone_optimizer.step()

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

        local_result = [self.model_parameters, data_size]
        return local_result
    
    def evaluate(self, payload):
        model_parameters = payload[0]
        self.set_model(model_parameters)
        
        model = nn.Sequential(self.backbone, self.model)
        model = model.eval()

        test_loader = self.dataset.get_dataloader(train=False, batch_size=128)

        loss_, acc_ = misc.evaluate(
            model,
            self.criterion,
            test_loader,
            self.device,
            multimodel=False,
            k=1,
        )

        return loss_, acc_
    
    def noisy_eval(self,payload):
        # model_parameters = payload[0]
        # self.set_model(model_parameters)
        
        model = nn.Sequential(self.backbone, self.model)
        model = model.eval()

        train_loader = self.dataset.get_dataloader(cid=self.g_cid, batch_size=128, train=True)
        train_metrics = self.validate(model, self.device, train_loader, logger=None)
        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                {
                    f"client-{self.g_cid}/loss": train_metrics["overall_noisy_test_loss"],
                    f"client-{self.g_cid}/acc": train_metrics["overall_noisy_accuracy"],
                    f"client-{self.g_cid}/clean_acc": train_metrics["clean_accuracy"],
                    f"client-{self.g_cid}/noisy_acc": train_metrics["noisy_accuracy"],
                    f"client-{self.g_cid}/noisy_fit_acc": train_metrics["noisy_fit_accuracy"], 
                },
                commit=False,
            )

        model_parameters = payload[0]
        self.set_model(model_parameters)
        
        model = nn.Sequential(self.backbone, self.model)
        model = model.eval()
        train_metrics = self.validate(model, self.device, train_loader, logger=None)
        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                {
                    f"g-client-{self.g_cid}/loss": train_metrics["overall_noisy_test_loss"],
                    f"g-client-{self.g_cid}/acc": train_metrics["overall_noisy_accuracy"],
                    f"g-client-{self.g_cid}/clean_acc": train_metrics["clean_accuracy"],
                    f"g-client-{self.g_cid}/noisy_acc": train_metrics["noisy_accuracy"],
                    f"g-client-{self.g_cid}/noisy_fit_acc": train_metrics["noisy_fit_accuracy"], 
                },
                commit=False,
            )

    
    def nll_monitor(self, payload, logger):
        model_parameters = payload[0]
        self.set_model(model_parameters)
        
        model = nn.Sequential(self.backbone, self.model)
        model = model.eval()

        train_loader = self.dataset.get_overall_dataloader(batch_size=128)

        train_metrics = self.validate(model, self.device, train_loader, logger)

        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                {
                    "train/loss": train_metrics["overall_noisy_test_loss"],
                    "train/acc": train_metrics["overall_noisy_accuracy"],
                    "train/clean_acc": train_metrics["clean_accuracy"],
                    "train/noisy_acc": train_metrics["noisy_accuracy"],
                    "train/noisy_fit_acc": train_metrics["noisy_fit_accuracy"], 

                    "gmm/f1": train_metrics["f1"],
                    "gmm/precision": train_metrics["precision"],
                    "gmm/recall": train_metrics["recall"],
                    "gmm/accuracy": train_metrics["accuracy"],
                },
                commit=False,
            )

    def validate(self, model, device, data_loader, logger):
        model.eval()
        overall_clean_loss = 0
        overall_noisy_loss = 0
        clean_set_correct = 0
        noisy_set_correct = 0
        clean_correct = clean_cnt = 1e-8
        noisy_set_overfitted = 0
        noisy_cnt = 1e-8
        losses = []
        y_true = []
        with torch.no_grad():
            for data in data_loader:
                data, target, noisy_target = data["img"].to(device), data["label"].to(device), data["noisy_label"].to(device)  # send data to device
                output = model(data)  # forward
                clean_mask = target == noisy_target
                overall_clean_loss += TF.cross_entropy(output, target, reduction="sum").item()  # compute loss
                overall_noisy_loss += TF.cross_entropy(output, noisy_target, reduction="sum").item()  # compute loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                noisy_set_correct += pred[~clean_mask].eq(target[~clean_mask].view_as(pred[~clean_mask])).sum().item()  # count correct predictions
                clean_set_correct += pred[clean_mask].eq(target[clean_mask].view_as(pred[clean_mask])).sum().item()  # count correct predictions
                noisy_set_overfitted += pred[~clean_mask].eq(noisy_target[~clean_mask].view_as(pred[~clean_mask])).sum().item()
                clean_cnt += torch.sum(clean_mask).item()
                noisy_cnt += torch.sum(~clean_mask).item()
                losses.append(TF.cross_entropy(output, target, reduction="none").cpu().numpy())
                y_true.append(clean_mask.cpu().numpy())
        losses = np.concatenate(losses)
        y_true = np.concatenate(y_true)
        overall_clean_loss /= len(data_loader.dataset)
        overall_noisy_loss /= len(data_loader.dataset)
        overall_clean_accuracy = (clean_set_correct+noisy_set_correct) / len(data_loader.dataset)
        overall_noisy_accuracy = (noisy_set_overfitted+clean_set_correct) / len(data_loader.dataset)
        clean_set_acc = clean_set_correct / len(data_loader.dataset)
        noisy_set_acc = noisy_set_correct / len(data_loader.dataset)
        noisy_fit_acc = noisy_set_overfitted / len(data_loader.dataset)
        if logger is not None:
            logger.info(
                f"Round [{self.round}/{self.args.com_round}] "
                f"Avg loss: {overall_clean_loss:.4f}, Avg N loss: {overall_noisy_loss:.4f}, "
                f"C set acc: {clean_set_correct:.0f}/{clean_cnt:.0f}/{len(data_loader.dataset)}({clean_set_correct/clean_cnt*100:.2f}%,{clean_set_correct/len(data_loader.dataset)*100:.2f}%), "
                f"N set acc: {noisy_set_correct:.0f}/{noisy_cnt:.0f}/{len(data_loader.dataset)}({noisy_set_correct/noisy_cnt*100:.2f}%,{noisy_set_correct/len(data_loader.dataset)*100:.2f}%), "
                f"N set fit: {noisy_set_overfitted:.0f}/{noisy_cnt:.0f}/{len(data_loader.dataset)}({noisy_set_overfitted/noisy_cnt*100:.2f}%,{noisy_set_overfitted/len(data_loader.dataset)*100:.2f}%), "
                f"OC acc: {clean_set_correct+noisy_set_correct:.0f}/{len(data_loader.dataset)}({overall_clean_accuracy*100:.2f}%), "
                f"ON acc: {noisy_set_overfitted+clean_set_correct:.0f}/{len(data_loader.dataset)}({overall_noisy_accuracy*100:.2f}%)"
            )
            gmm_metrics = self.gmm(losses, y_true, logger)
            return {
                "overall_noisy_accuracy": overall_noisy_accuracy*100,
                "overall_noisy_test_loss": overall_noisy_loss,

                "overall_clean_accuracy": overall_clean_accuracy*100,
                "overall_clean_test_loss": overall_clean_loss,

                "clean_accuracy": clean_set_acc*100,
                "noisy_accuracy": noisy_set_acc*100,
                "noisy_fit_accuracy": noisy_fit_acc*100,
                
                **gmm_metrics,
            }
        else:
            gmm_metrics = None
            return {
                "overall_noisy_accuracy": overall_noisy_accuracy*100,
                "overall_noisy_test_loss": overall_noisy_loss,

                "overall_clean_accuracy": overall_clean_accuracy*100,
                "overall_clean_test_loss": overall_clean_loss,

                "clean_accuracy": clean_set_acc*100,
                "noisy_accuracy": noisy_set_acc*100,
                "noisy_fit_accuracy": noisy_fit_acc*100,
            }
        

    def gmm(self, gmm_metrics, y_true, logger):
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        gmm.fit(gmm_metrics.reshape((-1, 1)))
        prob = gmm.predict_proba(gmm_metrics.reshape((-1, 1)))
        prob = prob[:,gmm.means_.argmin()] #属于小loss的概率是多少 获得的是真实无噪声样本的概率是多少 该样本无噪声的概率是多少
        y_pred = prob > 0.5
        gmm_metrics = {
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }
        logger.info(
            f"Round [{self.round}/{self.args.com_round}] "
            f"GMM f1: {gmm_metrics['f1']:.4f}, precision: {gmm_metrics['precision']:.4f}, recall: {gmm_metrics['recall']:.4f}, accuracy: {gmm_metrics['accuracy']:.4f}"
        )
        return gmm_metrics

# for warmup analysis, todo
class NLLFedProxClientTrainer(NLLFedAvgClientTrainer):
    def __init__(
        self,
        backbone,
        model,
        num_clients,
        cuda=True,
        device=None,
        logger=None,
        wandb_logger: WandbLogger=None,
        personal=False,
        args=None,
    ) -> None:
        super().__init__(
            backbone, model, num_clients, cuda, device, logger, wandb_logger, personal, args
        )

        self.mus = [1.0 for i in range(self.num_clients)]

    # adjust the mu for each client, if the training loss decrease in consecutive 5 rounds, then decrease the mu by 0.1 otherwise increase it by 0.1


    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum)

        model = nn.Sequential(self.backbone, self.model)
        model = model.train()

        frz_model = deepcopy(model)
        frz_model.eval()

        data_size = len(train_loader.dataset)

        self.on_client_training_start()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if not self.none_backbone:
            if self.backbone_lr_scheduler is not None:
                self.backbone_lr_scheduler.step()

        loss_ = AverageMeter()
        for epoch in range(self.epochs):
            self.on_training_epoch_start()

            model.train()
            
            loss_.reset()

            for batch in train_loader:
                self.on_training_batch_start()
                
                imgs, labels, noisy_labels, guids = batch["img"], batch["label"], batch["noisy_label"], batch["guid"]
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = model(imgs)
                loss = self.criterion(outputs, noisy_labels)
                
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                mu = 1.0
                loss = loss + 0.5 * mu * l2
                
                self.optimizer.zero_grad()
                if not self.none_backbone:
                    self.backbone_optimizer.zero_grad()
                model.zero_grad()

                loss.backward()

                self.on_optimize_step_start()
                
                self.optimizer.step()
                if not self.none_backbone:
                    self.backbone_optimizer.step()

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

        local_result = [self.model_parameters, data_size]
        return local_result