import torch
import torch.nn as nn
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
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)


class FedProxLocalLossMeterHook(SerialClientTrainerHook):
    def __init__(self, args):
        self.loss_meter = [AverageMeter()] * args.num_clients
    
    def on_client_training_start(self, client_trainer, *args, **kwargs):
        self.loss_meter[client_trainer.l_cid].reset()

    def update(self, client_trainer, l_cid, batch_loss, *args, **kwargs):
        self.loss_meter[l_cid].update(batch_loss)

    def get_loss_avg(self, client_trainer, l_cid, *args, **kwargs):
        return self.loss_meter[l_cid].avg


class FedProxGlobalAdaptiveMuScheduler(SyncServerHook):
    def __init__(self, args, init_mu: float = None, patience: int = 5, mu_delta: float = 0.1, max_mu: float = 1., min_mu: float = 0.):
        super(FedProxGlobalAdaptiveMuScheduler, self).__init__()
        self.mu = init_mu
        if self.mu is None:
            self.mu = 1. if args.partition == "iid" else 0.
        self.patience = patience
        self.prev_losses = float("inf")
        self.patience_cnt = 0
        self.mu_delta = mu_delta
        self.max_mu = max_mu
        self.min_mu = min_mu

    def local_losses_avg(self, local_losses, weights, *args, **kwargs):
        return sum([local_losses[i] * weights[i] for i in range(len(local_losses))]) / sum(weights)

    def step(self, server_handler, local_losses, weights, *args, **kwargs):
        loss = self.local_losses_avg(local_losses, weights)
        if loss < self.prev_losses:
            self.patience_cnt += 1
            if self.patience_cnt >= self.patience:
                self.mu -= self.mu_delta
                self.mu = max(self.min_mu, self.mu)
                self.patience_cnt = 0
        else:
            self.mu += self.mu_delta
            self.mu = min(self.max_mu, self.mu)
        self.prev_losses = loss
    
    def get_mu(self, server_handler, *args, **kwargs):
        return self.mu


class FedProxMuConstantScheduler(SyncServerHook):
    def __init__(self, args, init_mu: float = 0.1):
        self.mu = init_mu
        
    def step(self, server_handler, *args, **kwargs):
        pass

    def get_mu(self, server_handler, *args, **kwargs):
        return self.mu