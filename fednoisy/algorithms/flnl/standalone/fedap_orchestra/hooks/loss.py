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
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)


class SupOrchestraLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs_s, targets, *args, **kwargs):
        sup_loss = TF.cross_entropy(outputs_s["linear_head"], targets["linear_head"])
        
        orchestra_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs_s["orchestra_head"]+1e-10, dim=1), dim=1).mean()

        loss = sup_loss + orchestra_loss
        return loss
    

class SemiOrchestraLoss(SupOrchestraLoss):
    def loss(self, client_trainer, outputs_w, outputs_s, targets, *args, **kwargs):
        with torch.no_grad():
            probs, preds = torch.max(torch.softmax(outputs_w["linear_head"], dim=1), dim=1)
            mask = (probs > 0.9)
        
        sup_targets = targets["linear_head"]
        sup_targets[mask] = preds[mask]
        sup_loss = TF.cross_entropy(outputs_s["linear_head"], sup_targets)

        orchestra_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs_s["orchestra_head"]+1e-10, dim=1), dim=1).mean()
        loss = sup_loss + orchestra_loss
        return loss