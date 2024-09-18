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


class ClientCheckPointHook(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()


class ServerCheckPointHook(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()