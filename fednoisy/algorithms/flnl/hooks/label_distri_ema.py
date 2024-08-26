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


class ClientLabelDistriEMA(SerialClientTrainerHook):
    def __init__(self, alpha: float = 0.8) -> None:
        super().__init__()
        self.alpha = alpha

    def on_init(self, client_trainer, *args, **kwargs):
        self.label_distris = [None] * client_trainer.num_clients

    def is_init(self, label_distri):
        return label_distri is None

    @torch.no_grad()
    def label_distri_ema(self, client_trainer, outputs, *args, **kwargs):
        # update the label distribution
        prob = torch.softmax(outputs, dim=1).mean(dim=0)

        label_distri = self.label_distris[client_trainer.l_cid]
        if self.is_init(label_distri):
            label_distri = prob
        else:        
            label_distri = self.alpha * label_distri + (1 - self.alpha) * prob

    def get_label_distri(self, client_trainer, cid, *args, **kwargs):
        return self.label_distris[cid]