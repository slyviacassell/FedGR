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

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)


class LocalKNNMonitor(SerialClientTrainerHook):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    @torch.no_grad()
    def local_train_knn_monitor(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        model = client_trainer.model
        model.eval()
        feats = []
        labels = []
        clean_mask = []
        for batch in dataloader:
            imgs, targets, noisy_targets = batch["img"], batch["label"], batch["noisy_label"]
            imgs = imgs.to(client_trainer.device)
            feat = model.get_embedding(imgs)
            feats.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            clean_mask.append(noisy_targets.numpy() == targets.numpy())
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        model.train()

        # KNN classifier
        neigh = KNeighborsClassifier(n_neighbors=self.k, metric="cosine", n_jobs=12)
        neigh.fit(feats, labels)
        preds = neigh.predict(feats)
        acc = accuracy_score(labels, preds)
        print(
            f"Round {client_trainer.round_idx} "
            f"KNN: {acc*100:.2f}%"
        )


# todo
class LocalKNNClassifier(SerialClientTrainerHook):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    @torch.no_grad()
    def local_train_knn_classifier(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        model = client_trainer.model
        model.eval()
        feats = []
        labels = []
        clean_mask = []
        for batch in dataloader:
            imgs, targets, noisy_targets = batch["img"], batch["label"], batch["noisy_label"]
            imgs = imgs.to(client_trainer.device)
            feat = model.get_embedding(imgs)
            feats.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            clean_mask.append(noisy_targets.numpy() == targets.numpy())
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        model.train()

        
