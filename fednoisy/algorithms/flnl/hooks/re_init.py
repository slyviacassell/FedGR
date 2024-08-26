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

from fednoisy.models.build_model import build_model, build_multi_model
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    NORM_VALUES,
)

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)


class ReInitNetworkHook(SyncServerHook):
    def __init__(self, interval: int = 100) -> None:
        super().__init__()
        self.interval = interval

    def on_global_update_end(self, server, *args, **kwargs):
        if self.every_n_round(server, self.interval):
            server._LOGGER.info(f"Reinit model at round {server.round}")
            model = build_model(server.args.model, CLASS_NUM[server.args.dataset], dataset=server.args.dataset)
            model.to(server.device)
            state_dict = model.state_dict()
            server.model.load_state_dict(state_dict)