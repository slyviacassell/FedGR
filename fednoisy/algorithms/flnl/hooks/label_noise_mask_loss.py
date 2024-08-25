import torch
import torch.nn as nn
import torch.nn.functional as TF
import math
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, auc, precision_score, f1_score
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)


class LabelNoiseMaskOutLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
        self.p_meter = AverageMeter()
        self.meter = AverageMeter()

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(f"Round {client_trainer.round}: p_meter: {self.p_meter.avg*100:.2f}% | meter: {self.meter.avg*100:.2f}%")
        self.p_meter.reset()
        self.meter.reset()

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if client_trainer.round == client_trainer.args.warmup_round:
            self.overall_noisy_guids = client_trainer.overall_noisy_guids.copy()

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        labels = kwargs["labels"].numpy()
        p_acc = labels == targets.cpu().numpy()
        self.meter.update(p_acc.mean(), len(p_acc))

        assert client_trainer.args.warmup_round > 1
        if client_trainer.round < client_trainer.args.warmup_round:
            loss = client_trainer.criterion(outputs, targets) # default loss
        else:
            guids = kwargs["guids"].numpy()
            # c_mask = np.array([False if g in self.overall_noisy_guids else True for g in guids]) # mask out noisy samples
            
            # p_mask = np.array([True if g in client_trainer.p_labels.index else False for g in guids])
            # p_mask = (~c_mask)&p_mask
            # if p_mask.sum() != 0:
            #     p_labels = client_trainer.p_labels.loc[guids[p_mask]]["freqent_pred"].tolist()
            #     p_labels = torch.tensor(p_labels).to(client_trainer.device)
            #     targets[p_mask] = p_labels

            # if c_mask.sum() != len(c_mask):
            #     targets[~c_mask][:5] = torch.tensor(labels[~c_mask][:5]).to(client_trainer.device)

            # c_mask = torch.tensor(c_mask).to(client_trainer.device)
            loss = TF.cross_entropy(outputs, targets, reduction="none")
            # loss = (loss * c_mask).mean()
            # loss = (loss * c_mask).sum() / (c_mask.sum() + 1e-8)
            loss = loss.mean()

            # if client_trainer.hooks_dict.get("label_distri_ema", None) is not None:
            #     label_distri = client_trainer.call_hook("get_label_distri", "label_distri_ema", client_trainer.l_cid)
            #     if label_distri is not None:
            #         outputs = torch.softmax(outputs, dim=1)
            #         penalty = -torch.sum(label_distri * torch.log(outputs + 1e-8), dim=1)
            #         loss += penalty.mean()

        p_acc = labels == targets.cpu().numpy()
        self.p_meter.update(p_acc.mean(), len(p_acc))

        return loss


class LabelNoiseOrcaleMaskOutLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        assert client_trainer.args.warmup_round > 1
        if client_trainer.round < client_trainer.args.warmup_round:
            loss = client_trainer.criterion(outputs, targets)
        else:
            mask = kwargs["mask"] # tensor mask
            loss = TF.cross_entropy(outputs, targets, reduction="none")
            loss = (loss * mask).mean()
        
        return loss
    

class LabelNoiseTruncationLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        client_trainer.e_ratios = [client_trainer.dataset.get_dataset(cid).get_noise_rate() for cid in range(client_trainer.args.num_clients)]

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        assert client_trainer.args.warmup_round > 1
        if client_trainer.round < client_trainer.args.warmup_round:
            loss = client_trainer.criterion(outputs, targets)
        else:
            loss = TF.cross_entropy(outputs, targets, reduction="none")
            mask_out_ratio = client_trainer.e_ratios[client_trainer.l_cid]
            # mask out top-k noisy samples
            mask = torch.ones_like(loss)
            mask_out_num = int(mask_out_ratio * len(mask))
            _, indices = loss.topk(mask_out_num)
            mask[indices] = 0
            loss = (loss * mask).sum()/(mask.sum() + 1e-8)
        
        return loss
    