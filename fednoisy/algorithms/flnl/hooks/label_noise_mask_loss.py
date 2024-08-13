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


class LabelNoiseMaskOutLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        assert client_trainer.args.warmup_round > 1
        if client_trainer.round < client_trainer.args.warmup_round:
            loss = client_trainer.criterion(outputs, targets) # default loss
        else:
            guids = kwargs["guids"].numpy()
            # mask = [True if g in client_trainer.overall_clean_guids else False for g in guids] # select only clean samples
            mask = [False if g in client_trainer.overall_noisy_guids else True for g in guids] # mask out noisy samples
            mask = torch.tensor(mask).to(client_trainer.device)
            loss = TF.cross_entropy(outputs, targets, reduction="none")
            loss = (loss * mask).mean()
            # loss = (loss * mask).sum() / (mask.sum() + 1e-8)

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
    