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
class LocalSniffer(SerialClientTrainerHook):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

        self.df_list = []

    @torch.no_grad()
    def process(self, model, dataloader, device, *args, **kwargs):
        
        model.eval()
        feats = []
        confis = []
        labels = []
        clean_mask = []
        entropies = []
        losses = []
        for batch in dataloader:
            img, target, noisy_target = batch["img"], batch["label"], batch["noisy_label"]
            img = img.to(device)
            noisy_target = noisy_target.to(device)
            feat = model(img, enable_encoder=True, enable_decoder=False)
            logit = model(feat, enable_encoder=False, enable_decoder=True)
            confi = TF.softmax(logit, dim=1)
            loss = TF.cross_entropy(logit, noisy_target, reduction="none")
            
            feats.append(feat.cpu().numpy())
            confis.append(confi.cpu().numpy())
            labels.append(target.numpy())
            clean_mask.append(noisy_target.cpu().numpy() == target.numpy())
            entropies.append(-torch.sum(confi * torch.log(confi + 1e-10), dim=1).cpu().numpy())
            losses.append(loss.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        confis = np.concatenate(confis, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        entropies = np.concatenate(entropies, axis=0)
        losses = np.concatenate(losses)
        model.train()

        metric = losses 

        return metric, clean_mask
    
    def on_client_training_start(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128) 
        model = client_trainer.model
        # model = client_trainer.local_ema_models[client_trainer.l_cid]
        metric,clean_mask = self.process(model, dataloader, client_trainer.device)
        df = pd.DataFrame({"metric": metric, "clean_mask": clean_mask})
        self.df_list.append(df)
    
    def on_local_process_end(self, client_trainer, *args, **kwargs):
        df = pd.concat(self.df_list)
        # plot the loss distribution according to clean_mask
        plt.figure()
        sns.histplot(df, x="metric", hue="clean_mask", kde=True)
        plt.savefig(f"metric_distribution_{client_trainer.round:03}.png")
        plt.close()
        self.df_list = []

    @torch.no_grad()
    def process_v2(self, model, dataloader, device, *args, **kwargs):
        
        model.eval()
        feats = []
        confis = []
        labels = []
        clean_mask = []
        entropies = []
        losses = []
        for batch in dataloader:
            img, target, noisy_target = batch["img"], batch["label"], batch["noisy_label"]
            img = img.to(device)
            feat = model(img, enable_encoder=True, enable_decoder=False)
            logit = model(feat, enable_encoder=False, enable_decoder=True)
            confi = TF.softmax(logit, dim=1)
            loss = TF.cross_entropy(logit, noisy_target)
            
            feats.append(feat.cpu().numpy())
            confis.append(confi.cpu().numpy())
            labels.append(target.numpy())
            clean_mask.append(noisy_target.numpy() == target.numpy())
            entropies.append(-torch.sum(confi * torch.log(confi + 1e-10), dim=1).cpu().numpy())
            losses.append(loss.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        confis = np.concatenate(confis, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        entropies = np.concatenate(entropies, axis=0)
        losses = np.concatenate(losses, axis=0)
        model.train()

        metric = losses + entropies

        return metric, clean_mask
