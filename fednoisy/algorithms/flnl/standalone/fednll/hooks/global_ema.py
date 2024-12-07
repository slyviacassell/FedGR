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
from fednoisy.utils.ema import EMA


class GlobalEMAHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.anchor_model = EMA(
                client_trainer.cur_global_model,
                beta=0.99,
                update_after_step=0,
                update_every=1,
                inv_gamma=1.0,
                power=1.0,
            )
        client_trainer.anchor_model.initted.data.copy_(torch.Tensor([True]))

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.update(client_trainer)

    def update(self, client_trainer, *args, **kwargs):
        client_trainer.anchor_model.update()