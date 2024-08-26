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

from fednoisy.utils.misc import (
    AverageMeter,
)

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)


class LabelNoiseMonitor(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_global_update_end(self, handler, *args, **kwargs):
        noisy_guids = handler.noisy_guids
        n_clients = handler.args.num_clients
        t_ratios = []
        e_ratios = []
        for cid in range(n_clients):
            t_ratio = handler.dataset.get_dataset(cid).get_noise_rate()
            guids = handler.dataset.get_dataset(cid).guids
            e_ratio = [1 if g in noisy_guids else 0 for g in guids]
            e_ratio = sum(e_ratio) / len(e_ratio)
            t_ratios.append(t_ratio)
            e_ratios.append(e_ratio)

        # calculate the pearson coefficient between t_ratios and e_ratios
        pearson = np.corrcoef(t_ratios, e_ratios)[0, 1]
        t_str = ",".join([f"{t:.3f}" for t in t_ratios])
        e_str = ",".join([f"{e:.3f}" for e in e_ratios])
        handler._LOGGER.info(
            f"Round {handler.round}:\n"
            f"T: {t_str}\n"
            f"E: {e_str}\n"
            f"Pearson: {pearson:.4f}"
        )