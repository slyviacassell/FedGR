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

from sklearn.neighbors import KNeighborsClassifier
import scipy
from sklearn.utils.extmath import weighted_mode

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)


class WeightSchedulerHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.est_cid_noise = [0] * client_trainer.args.num_clients
        self.cid_n_samples = [0] * client_trainer.args.num_clients

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.get_est_cid_noise(client_trainer)

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            self.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

            self.cid_n_samples[cid] = len(n_mask)

    def step(self, client_trainer, sl_weight, *args, **kwargs):
        raise NotImplementedError
    

class SLWeightSchedulerHook(WeightSchedulerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.decay_round = 50
        super().on_init(client_trainer, *args, **kwargs)
    
    def step(self, client_trainer, sl_weight, *args, **kwargs):
        if client_trainer.round < client_trainer.args.sniffing_round:
            weight = sl_weight
        elif client_trainer.round < client_trainer.args.sniffing_round + client_trainer.args.warmup_round:
            weight = sl_weight * (1. - self.est_cid_noise[client_trainer.l_cid]) * min(1., (client_trainer.round - client_trainer.args.sniffing_round) / min(client_trainer.args.warmup_round, self.decay_round))

            # weight = sl_weight
        else:
            # c_size = len(client_trainer.hooks_dict["sl_loss"].relabels[client_trainer.g_cid])
            # d_size = len(client_trainer.dataset.get_dataset(client_trainer.g_cid))
            # n_rate = self.est_cid_noise[client_trainer.g_cid]
            # weight = sl_weight * (c_size / d_size)
            
            weight = sl_weight
        return weight
    

class SSLWeightSchedulerHook(WeightSchedulerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.decay_round = 50
        super().on_init(client_trainer, *args, **kwargs)
    
    def step(self, client_trainer, ssl_weight, *args, **kwargs):
        if client_trainer.round < client_trainer.args.sniffing_round:
            weight = ssl_weight
        elif client_trainer.round < client_trainer.args.sniffing_round + client_trainer.args.warmup_round:
            # weight = ssl_weight * self.est_cid_noise[client_trainer.l_cid] * min(1., (client_trainer.round - client_trainer.args.sniffing_round) / min(client_trainer.args.warmup_round, self.decay_round))
            
            weight = ssl_weight
        else:
            # c_size = len(client_trainer.hooks_dict["sl_loss"].relabels[client_trainer.g_cid])
            # d_size = len(client_trainer.dataset.get_dataset(client_trainer.g_cid))
            # n_rate = self.est_cid_noise[client_trainer.g_cid]
            # if n_rate < 0.2:
            #     weight = 0.
            # else:
            #     weight = ssl_weight * (1. - c_size / d_size)

            weight = ssl_weight
        return weight