import torch
import argparse
import sys
import os
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from sklearn.mixture import GaussianMixture
from collections import Counter

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
from fednoisy.utils.criterion import get_robust_loss, mixup_criterion, loss_coteaching, get_volmin_loss, NegEntropy, DivideMixSemiLoss, FedLCLoss, loss_coteaching_guessing
from fednoisy.utils.mixup import mixup_data
from fednoisy.utils import dynamic_bootstrapping as dynboot
from fednoisy.utils.lr_scheduler import get_lr_scheduler
from fednoisy.utils.wandb_logger import WandbLogger


class FedNLLFedAvgClientTrainer(SGDSerialClientTrainer):
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
        self.cache = []
        self.args = args
        self.wandb_logger = wandb_logger

    @property
    def model_parameters(self) -> torch.Tensor:
        return misc.serialize_model(self._model)

    def set_model(self, parameters: torch.Tensor):
        misc.deserialize_model(self._model, parameters)

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
    
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list, cur_round):
        self.round = cur_round
        self._LOGGER.info(f"Round {self.round} selected clients: {id_list}")
        model_parameters = payload[0]
        for cid in id_list:
            self.cur_cid = cid
            data_loader = self.dataset.get_dataloader(
                cid=cid, train=True, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)
            loss_, acc_ = self.evaluate()
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_*100:.2f}%, local test loss: {loss_:.4f}"
            )
            self.cache.append(pack)

        if self.wandb_logger is not None:
            self.wandb_logger.run.log({
                "global/lr": self.optimizer.param_groups[0]["lr"],
            })
        
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        eval_train_loader = self.dataset.get_eval_train_dataloader(self.args.dataset,cid=self.cur_cid)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            self._model.train()
            for batch in train_loader:
                imgs, labels, noisy_labels, guid = batch["img"], batch["label"], batch["noisy_label"], batch["guid"]
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, noisy_labels)
                
                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self.optimizer.step()

        loss_, acc_ = self.evaluate(self._model,eval_train_loader)
        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} local train {self.epochs} epochs accuracy: {acc_*100:.2f}%, loss: {loss_:.4f},"
        )

        if self.wandb_logger is not None:
            logs = {
                f"client-{self.cur_cid}/local-train-accuracy": acc_,
                f"client-{self.cur_cid}/local-train-loss": loss_,
            }
            self.wandb_logger.run.log(logs, commit=False)

        local_result = [self.model_parameters, data_size]
        return local_result

    def evaluate(self,model=None,data_loader=None):
        if model is not None and data_loader is not None:
            multimodel = hasattr(model, "models")
            loss_, acc_ = misc.evaluate(
                model,
                nn.CrossEntropyLoss(),
                data_loader,
                self.device,
                multimodel=multimodel,
            )
        else:
            test_loader = self.dataset.get_dataloader(train=False, batch_size=128)
            multimodel = hasattr(self._model, "models")
            loss_, acc_ = misc.evaluate(
                self._model,
                nn.CrossEntropyLoss(),
                test_loader,
                self.device,
                multimodel=multimodel,
            )

        return loss_, acc_
    

class FedNLLFedAvgMixupClientTrainer(FedNLLFedAvgClientTrainer):
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
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            wandb_logger,
            personal,
            args,
        )

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            train_loss = 0
            correct = 0
            total = 0
            batch_num = len(train_loader)
            for batch_idx, (imgs, labels, noisy_labels) in enumerate(train_loader):
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                imgs, targets_a, targets_b, lmbd = mixup_data(
                    imgs, noisy_labels, self.args.mixup_alpha, self.device
                )

                outputs = self.model(imgs)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lmbd
                )

                self.optimizer.zero_grad()
                self._model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # with torch.no_grad():
                #     train_loss += loss.detach()
                #     _, pred = torch.max(outputs.data, 1)
                #     total += noisy_labels.shape[0]
                #     correct += (
                #         lmbd * pred.eq(targets_a.data).cpu().sum().float()
                #         + (1 - lmbd) * pred.eq(targets_b.data).cpu().sum().float()
                #     )
            # avg_train_loss = train_loss / batch_num
            # train_acc = correct / total
            # self._LOGGER.info(
            #     f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] train accuracy: {train_acc*100:.2f}%; local train loss: {avg_train_loss:.2f}"
            # )

        local_result = [self.model_parameters, data_size]
        return local_result


class FedNLLFedAvgCoteachingClientTrainer(FedNLLFedAvgClientTrainer):
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
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            wandb_logger,
            personal,
            args,
        )

        # ---- initial hyperparameter setting ----
        # TODO: a possible hyperparameter setting for co-teaching in FL
        if args.coteaching_forget_rate is None:
            if args.globalize is True:
                estimate_noise_ratio = args.noise_ratio
            else:
                estimate_noise_ratio = (args.min_noise_ratio + args.max_noise_ratio) / 2
            self.coteaching_forget_rate = [
                estimate_noise_ratio for _ in range(num_clients)
            ]
        else:
            self.coteaching_forget_rate = [
                args.coteaching_forget_rate for _ in range(num_clients)
            ]

        rate_schedule = [
            np.ones(args.com_round) * self.coteaching_forget_rate[cid]
            for cid in range(num_clients)
        ]
        for cid in range(num_clients):
            rate_schedule[cid][: args.coteaching_num_gradual] = np.linspace(
                0,
                self.coteaching_forget_rate[cid] ** args.coteaching_exponent,
                args.coteaching_num_gradual,
            )
        self.rate_schedule = rate_schedule

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

    def train(self, model_parameters, train_loader):
        pure_ratio1_ = AverageMeter()
        pure_ratio2_ = AverageMeter()
        acc1_ = AverageMeter()
        acc2_ = AverageMeter()
        noisy_acc1_ = AverageMeter()
        noisy_acc2_ = AverageMeter()
        loss1_ = AverageMeter()
        loss2_ = AverageMeter()

        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.models[0].train()
        self._model.models[1].train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            for iter_idx, batch in enumerate(train_loader):
                imgs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
                noise_or_not = noisy_labels == labels
                batch_size = len(noisy_labels)
                if self.cuda:
                    imgs = imgs.to(self.device)
                    noisy_labels = noisy_labels.to(self.device)

                outputs = self.model(imgs)

                with torch.no_grad():
                    _, predicted1 = torch.max(outputs[0], 1)
                    _, predicted2 = torch.max(outputs[1], 1)
                    acc1_.update(
                        torch.sum(predicted1.cpu().eq(labels)).item() / batch_size,
                        batch_size,
                    )
                    acc2_.update(
                        torch.sum(predicted2.cpu().eq(labels)).item() / batch_size,
                        batch_size,
                    )
                    noisy_acc1_.update(
                        torch.sum(predicted1.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )
                    noisy_acc2_.update(
                        torch.sum(predicted2.eq(noisy_labels)).item() / batch_size,
                        batch_size,
                    )

                loss1, loss2, batch_pure_ratio1, batch_pure_ratio2 = loss_coteaching(
                    outputs[0],
                    outputs[1],
                    noisy_labels,
                    self.rate_schedule[self.cur_cid][self.round],
                    noise_or_not,
                )
                pure_ratio1_.update(100 * batch_pure_ratio1.item(), batch_size)
                pure_ratio2_.update(100 * batch_pure_ratio2.item(), batch_size)
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
                # self._LOGGER.info(
                #     f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] iter {iter_idx}: "
                #     f"loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}"
                # )

        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} local train done: "
            f"loss1={loss1_.avg:.4f}, loss2={loss2_.avg:.4f}, "
            f"train_acc1={acc1_.avg*100:.2f}%, train_acc2={acc2_.avg*100:.2f}%, "
            f"train_noisy_acc1={noisy_acc1_.avg*100:.2f}%, train_noisy_acc2={noisy_acc2_.avg*100:.2f}%, "
            f"pure_ratios1: {pure_ratio1_.avg:.2f}%, pure_ratios2: {pure_ratio2_.avg:.2f}%"
        )
        local_result = [self.model_parameters, data_size]
        return local_result


class FedNLLFedAvgDynamicBootstrappingClientTrainer(FedNLLFedAvgClientTrainer):
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
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            wandb_logger,
            personal,
            args,
        )

        # ---- initial hyperparameter setting ----
        # TODO: a possible hyperparameter setting for Dynamic Bootstraping in FL
        self.bmm_model = None
        self.bmm_model_maxLoss = 0
        self.bmm_model_minLoss = 0

        self.guidedMixup_round = int(105 / 300 * self.args.com_round)
        if args.dynboot_mixup == "dynamic":
            self.bootstrap_round_mixup = self.guidedMixup_round + int(
                5 / 300 * self.args.com_round
            )
        else:
            self.bootstrap_round_mixup = int(105 / 300 * self.args.com_round)

        self.temp_length = (
            int(200 / 300 * self.args.com_round) - self.bootstrap_round_mixup
        )
        self.temp_vec = np.linspace(1, 0.001, self.temp_length)
        self.k = 0

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.train()
        data_size = len(train_loader.dataset)

        for epoch in range(self.epochs):
            msg = ""
            first_flag = self.round * self.epochs + epoch == 0
            if self.args.dynboot_mixup == "static":
                alpha = self.args.dynboot_alpha
                if self.round < self.bootstrap_round_mixup:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: NORMAL mixup for {self.bootstrap_round_mixup} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp(
                        self._model, train_loader, self.optimizer, 32, self.device
                    )
                else:
                    if self.args.dynboot_bootbeta == "hard":
                        self._LOGGER.info(
                            f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: HARD BETA bootstrapping and NORMAL mixup from round {self.bootstrap_round_mixup+1}"
                        )
                        train_loss, train_acc = dynboot.train_mixUp_HardBootBeta(
                            self._model,
                            train_loader,
                            self.optimizer,
                            self.bmm_model,
                            self.bmm_model_maxLoss,
                            self.bmm_model_minLoss,
                            alpha,
                            self.args.dymboot_reg,
                            CLASS_NUM[self.args.dataset],
                            self.device,
                        )
                    elif self.args.dynboot_bootbeta == "soft":
                        self._LOGGER.info(
                            f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: SOFT BETA bootstrapping and NORMAL mixup from round {self.bootstrap_round_mixup+1}"
                        )
                        train_loss, train_acc = dynboot.train_mixUp_SoftBootBeta(
                            self._model,
                            train_loader,
                            self.optimizer,
                            self.bmm_model,
                            self.bmm_model_maxLoss,
                            self.bmm_model_minLoss,
                            alpha,
                            self.args.dymboot_reg,
                            first_flag,
                            self.device,
                        )

            if self.args.dynboot_mixup == "dynamic":
                alpha = self.args.dynboot_alpha
                if self.round < self.guidedMixup_round:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: NORMAL mixup for {self.guidedMixup_round} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp(
                        self._model, train_loader, self.optimizer, alpha, self.device
                    )
                elif self.round < self.bootstrap_round_mixup:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: Dynamic mixup from {self.guidedMixup_round} rounds"
                    )
                    train_loss, train_acc = dynboot.train_mixUp_Beta(
                        self.model,
                        train_loader,
                        self.optimizer,
                        self.bmm_model,
                        self.bmm_model_maxLoss,
                        self.bmm_model_minLoss,
                        alpha,
                        first_flag,
                        self.device,
                    )
                else:
                    self._LOGGER.info(
                        f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from {self.bootstrap_round_mixup} rounds"
                    )
                    k = min(
                        self.round - self.bootstrap_round_mixup,
                        self.temp_length - 1,
                    )
                    Temp = self.temp_vec[k]
                    train_loss, train_acc = dynboot.train_mixUp_SoftHardBetaDouble(
                        self._model,
                        train_loader,
                        self.optimizer,
                        self.bmm_model,
                        self.bmm_model_maxLoss,
                        self.bmm_model_minLoss,
                        alpha,
                        self.args.dynboot_reg,
                        first_flag,
                        Temp,
                        CLASS_NUM[self.args.dataset],
                        self.device,
                    )
                    msg = f"Temperature: {Temp:.4f}"

            # training tracking loss
            (
                self.bmm_model,
                self.bmm_model_maxLoss,
                self.bmm_model_minLoss,
            ) = dynboot.track_training_loss(self.model, train_loader, self.device)

            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]: train_acc: {train_acc*100:.2f}%, train_loss: {train_loss:.4f}, {msg}"
            )

        local_result = [self.model_parameters, data_size]
        return local_result

class FedNLLFedAvgDivideMixClientTrainer(FedNLLFedAvgClientTrainer):
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
        FedNLLFedAvgClientTrainer.__init__(
            self,
            model,
            num_clients,
            cuda,
            device,
            logger,
            wandb_logger,
            personal,
            args,
        )

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

    def local_process(self, payload, id_list, cur_round):
        self.round = cur_round
        self._LOGGER.info(f"Round {self.round} selected clients: {id_list}")
        model_parameters = payload[0]
        for cid in id_list:
            self.cur_cid = cid
            data_loader = self.dataset.get_dataloader(
                cid=cid, train=True, batch_size=self.batch_size
            )
            
            if cur_round < self.args.dividemix_warmup_round:
                pack = self.warmup(model_parameters, data_loader)
            else:
                pack = self.train(model_parameters, data_loader)

            pack.append(cid)

            loss_, acc_ = self.evaluate()

            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local test accuracy: {acc_*100:.2f}%, local test loss: {loss_:.4f}"
            )
            self.cache.append(pack)

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self.setup_optim(
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.momentum
        )
        self._model.models[0].train()
        self._model.models[1].train()

        loss_history1 = []
        loss_history2 = []

        data_size = len(train_loader.dataset)

        criterion = DivideMixSemiLoss()

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}]"
            )
            prob_dict1, label_guids1, unlabel_guids1 = self.update_probabilties_split_data_indices(self._model.models[0], loss_history1, train_loader)
            prob_dict2, label_guids2, unlabel_guids2 = self.update_probabilties_split_data_indices(self._model.models[1], loss_history2, train_loader)

            if len(label_guids2) == 0 or len(unlabel_guids2) == 0: # when gmm failed to find any labeled or unlabeled samples, simply warmup
                print('gmm f@cked',len(label_guids2), len(unlabel_guids2))
                self.warmup(model_parameters, train_loader)
            else:

                labeled_loader1 = self.dataset.get_dividemix_dataloader(
                    cid=self.cur_cid, train=True, batch_size=self.batch_size, selected_guid=label_guids2, sample_prob=prob_dict2 
                )
                unlabeled_loader1 = self.dataset.get_dividemix_dataloader(
                    cid=self.cur_cid, train=True, batch_size=self.batch_size, selected_guid=unlabel_guids2, sample_prob=prob_dict2
                )
                self.divide_mix(self.round, self._model.models[0],self._model.models[1],self.optimizer1,labeled_loader1,unlabeled_loader1,criterion,0)

            if len(label_guids1) == 0 or len(unlabel_guids1) == 0:
                print('gmm f@cked',len(label_guids1), len(unlabel_guids1))
                self.warmup(model_parameters, train_loader)
            else:

                labeled_loader2 = self.dataset.get_dividemix_dataloader(
                    cid=self.cur_cid, train=True, batch_size=self.batch_size, selected_guid=label_guids1, sample_prob=prob_dict1
                )
                unlabeled_loader2 = self.dataset.get_dividemix_dataloader(
                    cid=self.cur_cid, train=True, batch_size=self.batch_size, selected_guid=unlabel_guids1, sample_prob=prob_dict1
                )
                self.divide_mix(self.round, self._model.models[1],self._model.models[0],self.optimizer2,labeled_loader2,unlabeled_loader2,criterion,1)

        local_result = [self.model_parameters, data_size]
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
            # self._LOGGER.info(
            #     f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] iter {batch_idx}: "
            #     f"loss: {loss.item():.4f}"
            # )
        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} trained model {trained_model_idx} "
            f"loss: {loss_.avg:.4f}"
        )
     
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

        for epoch in range(self.epochs):
            self._LOGGER.info(
                f"Round {self.round} client-{self.cur_cid} local warmup epoch [{epoch}/{self.epochs}]"
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
                # self._LOGGER.info(
                #     f"Round {self.round} client-{self.cur_cid} local train epoch [{epoch}/{self.epochs}] iter {iter_idx}: "
                #     f"loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}"
                # )

        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} local train done: "
            f"loss1: {loss1_.avg:.4f}, loss2: {loss2_.avg:.4f}, "
            f"train_acc1={acc1_.avg*100:.2f}%, train_acc2={acc2_.avg*100:.2f}%,"
        )
        local_result = [self.model_parameters, data_size]
        return local_result
            
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
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        # Split data to labeled, unlabeled dataset
        pred = (prob > self.args.gmm_threshold)
        label_guids = pred.nonzero()[0]
        label_guids = indices[label_guids]

        self._LOGGER.info(
            f"Round {self.round} client-{self.cur_cid} dividemix gmm {self.args.gmm_threshold} "
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