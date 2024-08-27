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
        client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{client_trainer.g_cid} KNN: {acc*100:.2f}%")

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        self.local_train_knn_monitor(client_trainer, *args, **kwargs)


# todo
class LocalKNNClassifier(SerialClientTrainerHook):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

        self.df_list = []

    @torch.no_grad()
    def get_nn_confidence(self, model, dataloader, device, *args, **kwargs):
        
        model.eval()
        feats = []
        confis = []
        labels = []
        clean_mask = []
        for batch in dataloader:
            img, target, noisy_target = batch["img"], batch["label"], batch["noisy_label"]
            img = img.to(device)
            feat = model(img, enable_encoder=True, enable_decoder=False)
            logit = model(feat, enable_encoder=False, enable_decoder=True)
            confi = TF.softmax(logit, dim=1)
            
            feats.append(feat.cpu().numpy())
            confis.append(confi.cpu().numpy())
            labels.append(target.numpy())
            clean_mask.append(noisy_target.numpy() == target.numpy())
        feats = np.concatenate(feats, axis=0)
        confis = np.concatenate(confis, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        model.train()

        neigh = KNeighborsClassifier(n_neighbors=self.k, metric="cosine", n_jobs=12)
        neigh.fit(feats, labels)
        nn_confis = np.zeros_like(confis)
        for i in range(len(feats)):
            idx = neigh.kneighbors([feats[i]], return_distance=False) # (1, k)
            idx = idx[0]
            nn_confis[i] = np.mean(confis[idx], axis=0)

        nn_confis = nn_confis/np.max(nn_confis, axis=1, keepdims=True)
        loss = -np.sum(nn_confis * np.log(confis + 1e-10), axis=1)

        return loss, clean_mask
    
    def on_client_training_start(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        model = client_trainer.model
        loss,clean_mask = self.get_nn_confidence(model, dataloader, client_trainer.device)
        df = pd.DataFrame({"loss": loss, "clean_mask": clean_mask})
        self.df_list.append(df)
    
    def on_local_process_end(self, client_trainer, *args, **kwargs):
        df = pd.concat(self.df_list)
        # plot the loss distribution according to clean_mask
        plt.figure()
        sns.histplot(df, x="loss", hue="clean_mask", kde=True)
        plt.savefig(f"loss_distribution_{client_trainer.round}.png")
        plt.close()
        self.df_list = []

        
