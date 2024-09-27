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
import time

import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from collections import Counter
from fednoisy import data
from fednoisy.data.NLLData.functional import NoisyDataset
from fednoisy.utils.ema import EMA
from sklearn.mixture import GaussianMixture

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
from fednoisy.utils.criterion import get_robust_loss, NegEntropy, DivideMixSemiLoss
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
from fednoisy.algorithms.flnl.hooks import (
    FedProxLocalLossMeterHook,
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
        if self.args.use_fedprox:
            self.register_hooks(FedProxLocalLossMeterHook(self.args), "fedprox_loss_meter", "LOWEST")
        if type(self) == FedAPClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )
    
    def on_local_process_end(self, *args, **kwargs):
        super(FedAPClientTrainer, self).on_local_process_end(**kwargs)
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

        if self.is_prox:
            local_loss = self.call_hook("get_loss_avg", "fedprox_loss_meter", self.l_cid)
        else:
            local_loss = 0.
        local_loss = torch.tensor(local_loss)

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} local training done."
        )

        local_result = [self.model_parameters, data_size, local_loss, g_cid]
        return local_result
    

# speed up dividemix
class FedAPDivideMixClientTrainer(SGDSerialClientTrainer, SerialClientAlogrithmBase):
    def __init__(
            self, 
            model, 
            num_clients, 
            cuda=True, 
            device=None, 
            logger=None, 
            wandb_logger: WandbLogger = None, 
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

        self.cur_payload = None  

        self.is_prox = self.args.use_fedprox
        
        self.set_hooks() 

        self.on_init()

    def set_hooks(self):
        self.register_hooks(TestHook(test_interval=5), None, "LOWEST")
        if self.args.use_fedprox:
            self.register_hooks(FedProxLocalLossMeterHook(self.args), "fedprox_loss_meter", "LOWEST")
        if type(self) == FedAPDivideMixClientTrainer:
            self._LOGGER.info(
                f"Client Registered hooks: {self.hooks_dict.keys()}"
            )

    def on_local_process_end(self, *args, **kwargs):
        super(FedAPDivideMixClientTrainer, self).on_local_process_end(**kwargs)
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
    
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

    def set_global_cid(self, local_id_list, rank):
        global_id_list = local_id_list + (rank - 1) * self.num_clients
        self.global_id_list = global_id_list.tolist()

    def setup_optim(self, epochs, batch_size, lr, weight_decay, momentum):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer1 = torch.optim.SGD(
            self._model.models[0].parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        self.optimizer2 = torch.optim.SGD(
            self._model.models[1].parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CELoss = nn.CrossEntropyLoss()

    def local_process(self, payload, id_list, cur_round, rank):
        self.local_id_list = id_list.tolist()
        self.rank = rank.item()

        # get payloads
        self.cur_payload = payload
        self.round = cur_round.item()
        model_parameters = payload[0]  
        mu = payload[1]

        self.set_global_cid(id_list, rank)

        self._LOGGER.info(f"Round {self.round} selected clients global id: {self.global_id_list}, selected clients local id: {self.local_id_list}")

        self.on_local_process_start() # call self hook stage after get something
        
        # serial local training
        for l_cid, g_cid in zip(self.local_id_list, self.global_id_list):
            self.l_cid = l_cid
            self.g_cid = g_cid
            data_loader = self.dataset.get_dataloader(cid=self.g_cid, train=True, batch_size=self.batch_size)

            self.on_client_serial_process_start()

            if cur_round < self.args.dividemix_warmup_round:
                eval_loader = self.dataset.get_dataloader(
                    cid=g_cid, train=True, batch_size=self.batch_size
                )
                pack = self.warmup(model_parameters, data_loader)
            else:
                eval_loader = self.dataset.get_dataloader(
                    cid=g_cid, train=True, batch_size=128
                )
                pack = self.train(model_parameters, data_loader, eval_loader)

            self.cache.append(pack)

            self.on_client_serial_process_end()

        self.on_local_process_end()

    def train(self, model_parameters, train_loader, eval_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.models[0].train()
        self._model.models[1].train()

        loss_history1 = []
        loss_history2 = []

        data_size = len(train_loader.dataset)
        data_size = torch.tensor(data_size)
        g_cid = torch.tensor(self.g_cid)
        local_loss = torch.tensor(0.0)

        criterion = DivideMixSemiLoss()

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            prob_dict1, label_guids1, unlabel_guids1 = self.update_probabilties_split_data_indices(self._model.models[0], loss_history1, eval_loader)
            prob_dict2, label_guids2, unlabel_guids2 = self.update_probabilties_split_data_indices(self._model.models[1], loss_history2, eval_loader)

            if len(label_guids2) == 0 or len(unlabel_guids2) == 0: # when gmm failed to find any labeled or unlabeled samples, simply warmup
                print('gmm f@cked',len(label_guids2), len(unlabel_guids2))
                self.warmup(model_parameters, train_loader)
            else:
                labeled_loader1 = self.dataset.get_dividemix_dataloader(
                    cid=self.g_cid, train=True, batch_size=self.batch_size, selected_guid=label_guids2, sample_prob=prob_dict2, drop_last=False
                )
                unlabeled_loader1 = self.dataset.get_dividemix_dataloader(
                    cid=self.g_cid, train=True, batch_size=self.batch_size, selected_guid=unlabel_guids2, sample_prob=prob_dict2, drop_last=False
                )
                self.divide_mix(self.round, self._model.models[0],self._model.models[1],self.optimizer1,labeled_loader1,unlabeled_loader1,criterion,0)

            if len(label_guids1) == 0 or len(unlabel_guids1) == 0:
                print('gmm f@cked',len(label_guids1), len(unlabel_guids1))
                self.warmup(model_parameters, train_loader)
            else:

                labeled_loader2 = self.dataset.get_dividemix_dataloader(
                    cid=self.g_cid, train=True, batch_size=self.batch_size, selected_guid=label_guids1, sample_prob=prob_dict1, drop_last=False
                )
                unlabeled_loader2 = self.dataset.get_dividemix_dataloader(
                    cid=self.g_cid, train=True, batch_size=self.batch_size, selected_guid=unlabel_guids1, sample_prob=prob_dict1, drop_last=False
                )
                self.divide_mix(self.round, self._model.models[1],self._model.models[0],self.optimizer2,labeled_loader2,unlabeled_loader2,criterion,1)

        local_result = [self.model_parameters, data_size, local_loss, g_cid]
        return local_result

    def warmup(self, model_parameters, warmup_train_loader):
        '''
        warmup local trianing
        '''
        acc1_ = AverageMeter()
        acc2_ = AverageMeter()
        loss1_ = AverageMeter()
        loss2_ = AverageMeter()

        if self.args.noise_mode == 'asym' and not self.args.disable_asym_penalty:
            conf_penalty = NegEntropy()

        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.models[0].train()
        self._model.models[1].train()
        
        data_size = len(warmup_train_loader.dataset)
        data_size = torch.tensor(data_size)
        g_cid = torch.tensor(self.g_cid)
        local_loss = torch.tensor(0.0)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.g_cid} local warmup epoch [{epoch}/{self.epochs}]"
            )
            for iter_idx, batch in enumerate(warmup_train_loader):
                imgs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
                noise_or_not = noisy_labels == labels
                batch_size = len(noisy_labels)
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = self.model(imgs)

                if self.args.noise_mode == 'asym' and not self.args.disable_asym_penalty:
                    loss = [self.CELoss(o,noisy_labels) + conf_penalty(o) for o in outputs]
                else:
                    loss = [self.CELoss(o,noisy_labels) for o in outputs]
                loss1, loss2 = loss[0], loss[1]

                with torch.no_grad():
                    _, predicted1 = torch.max(outputs[0], 1)
                    _, predicted2 = torch.max(outputs[1], 1)
                    acc1_.update(
                        torch.sum(predicted1.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )
                    acc2_.update(
                        torch.sum(predicted2.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )
                    loss1_.update(loss1.item())
                    loss2_.update(loss2.item())

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self._model.models[0].zero_grad()
                self._model.models[1].zero_grad()
                loss1.backward()
                loss2.backward()
                self.optimizer1.step()
                self.optimizer2.step()

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} local train done: "
            f"loss1: {loss1_.avg:.4f}, loss2: {loss2_.avg:.4f}, "
            f"train_acc1={acc1_.avg*100:.2f}%, train_acc2={acc2_.avg*100:.2f}%,"
        )
        local_result = [self.model_parameters, data_size, local_loss, g_cid]
        return local_result
    
    def divide_mix(self,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion: DivideMixSemiLoss,trained_model_idx):
        loss_ = AverageMeter()
        net.train()
        net2.eval() #fix one network and train the other
        
        unlabeled_train_iter = iter(unlabeled_trainloader)    
        num_iter = (len(labeled_trainloader.dataset)//self.args.batch_size)+1
        for batch_idx, batch in enumerate(labeled_trainloader): 
            inputs_x, inputs_x2, labels_x, w_x = batch["img_0"], batch["img_1"], batch["noisy_label"], batch["weight"]
            try:
                u_batch = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                u_batch = unlabeled_train_iter.next()
            inputs_u, inputs_u2 = u_batch["img_0"], u_batch["img_1"]   
            batch_size = inputs_x.size(0)
            
            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, CLASS_NUM[self.args.dataset]).scatter_(1, labels_x.view(-1,1), 1)        
            w_x = w_x.view(-1,1).type(torch.FloatTensor) 

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(self.device), inputs_x2.to(self.device), labels_x.to(self.device), w_x.to(self.device)
            inputs_u, inputs_u2 = inputs_u.to(self.device), inputs_u2.to(self.device)

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)          
                
                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
                ptu = pu**(1/self.args.dividemix_temperature) # temparature sharpening
                
                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach()       
                
                # label refinement of labeled samples
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)            
                
                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x*labels_x + (1-w_x)*px              
                ptx = px**(1/self.args.dividemix_temperature) # temparature sharpening 
                        
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
                targets_x = targets_x.detach()       

            # mixmatch
            l = np.random.beta(self.args.dividemix_mixup_alpha, self.args.dividemix_mixup_alpha)        
            l = max(l, 1-l)
                    
            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
                    
            logits = net(mixed_input)
            logits_x = logits[:batch_size*2]
            logits_u = logits[batch_size*2:]        
            
            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:],self.args.dividemix_lambda_u, epoch+batch_idx/num_iter, self.args.dividemix_warmup_round)
            
            # regularization
            prior = torch.ones(CLASS_NUM[self.args.dataset])/CLASS_NUM[self.args.dataset]
            prior = prior.to(self.device)        
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = Lx + lamb * Lu  + penalty
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_.update(loss.item())
        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} trained model {trained_model_idx} "
            f"loss: {loss_.avg:.4f}"
        )
    
    def update_probabilties_split_data_indices(self, model, loss_history, train_loader):
        model.eval()
        losses_lst = []
        guid_lst = []
        clean_mask_lst = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets, guids, labels = batch["img"], batch["noisy_label"], batch["guid"], batch["label"]
                clean_mask = targets == labels
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                losses_lst.append(self.CE(outputs, targets))
                guid_lst.append(guids.numpy())
                clean_mask_lst.append(clean_mask.numpy())

        indices = np.concatenate(guid_lst)
        losses = torch.cat(losses_lst).cpu().numpy()
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        loss_history.append(losses)
        gmm_y_true = np.concatenate(clean_mask_lst)

        # Fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        # Split data to labeled, unlabeled dataset
        pred = (prob > self.args.gmm_threshold)
        label_guids = pred.nonzero()[0]
        label_guids = indices[label_guids]

        self._LOGGER.info(
            f"Round {self.round} client-{self.g_cid} dividemix gmm {self.args.gmm_threshold} "
            f"accuracy: {accuracy_score(gmm_y_true,pred)*100:.4f}%, "
            f"recall: {recall_score(gmm_y_true,pred)*100:.4f}%, "
            f"percision: {precision_score(gmm_y_true,pred)*100:.4f}%, "
            f"f1_score: {f1_score(gmm_y_true,pred)*100:.4f}%"
        )

        unlabel_guids = (~pred).nonzero()[0]
        unlabel_guids = indices[unlabel_guids]

        # Data index : probability
        prob_dict = {idx: prob for idx, prob in zip(indices, prob)}

        return prob_dict, label_guids, unlabel_guids