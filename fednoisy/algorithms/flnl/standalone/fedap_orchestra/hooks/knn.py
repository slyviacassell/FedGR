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
from scipy.spatial.distance import cdist

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)
from sklearn.manifold import TSNE
from fednoisy.utils.misc import (
    lid_term,
)
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
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


class LocalSniffer(SerialClientTrainerHook):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

        self.df_list = []
        self.relabels = {}

    @torch.no_grad()
    def process(self, client_trainer, model, dataloader, ema_model=None, *args, **kwargs):
        
        model.eval()
        feats = []
        confis = []
        labels = []
        noisy_labels = []
        clean_mask = []
        entropies = []
        sl_losses = []
        ssl_metrics = []
        topk_preds_list = []
        self_pseudo_labels = []
        for batch in dataloader:
            img, target, noisy_target, guids = batch["img"], batch["label"], batch["noisy_label"], batch["guid"]
            img = img.to(client_trainer.device)
            noisy_target = noisy_target.to(client_trainer.device)
            feat = model(img, enable_encoder=True, enable_decoder=False)
            outputs = model(feat, enable_encoder=False, enable_decoder=True, return_dict=True, full_heads=True)
            logit = outputs["linear_head"]["cls_head"]
            confi = TF.softmax(logit, dim=1)
            sl_loss = TF.cross_entropy(logit, noisy_target, reduction="none")

            # self labeling
            probs, preds = torch.max(confi, dim=1)
            max_confi_ema = client_trainer.max_confis_ema[client_trainer.l_cid]
            confi_ema = client_trainer.confis_ema[client_trainer.l_cid]
            confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            mask = (probs > confi_threshold[preds])
            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy()]).to(client_trainer.device)
            mask = mask & noisy_mask
            pseudo_labels = noisy_target.clone()
            pseudo_labels[mask] = preds[mask]

            topk_confis, topk_preds = torch.topk(confi, 2, dim=1)
            probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids.tolist()], device=client_trainer.device, dtype=torch.float32)
            probs_mask = probs > 0.99
            topk_preds[probs_mask] = noisy_target[probs_mask].unsqueeze(1).repeat(1, 2)

            bsz = img.size(0)
            if ema_model is not None:
                ema_outputs = ema_model(img, return_dict=True, full_heads=True)
                q = client_trainer.global_centroids(TF.normalize(ema_outputs["orchestra_head"], dim=1))
                q = TF.softmax(q / client_trainer.args.orchestra_temperature, dim=1)
                p = client_trainer.global_centroids(TF.normalize(outputs["orchestra_head"], dim=1))
                ssl_loss = -torch.sum(q * TF.log_softmax(p,dim=1), dim=1)
                ssl_entropy = -torch.sum(TF.softmax(p,dim=1) * TF.log_softmax(p,dim=1), dim=1)

            feats.append(feat.detach().cpu().numpy())
            confis.append(confi.detach().cpu().numpy())
            labels.append(target.numpy())
            noisy_labels.append(noisy_target.cpu().numpy())
            self_pseudo_labels.append(pseudo_labels.cpu().numpy())
            topk_preds_list.append(topk_preds.cpu().numpy())
            clean_mask.append(noisy_target.cpu().numpy() == target.numpy())
            entropies.append(-torch.sum(confi.detach() * torch.log(confi.detach() + 1e-10), dim=1).cpu().numpy())
            sl_losses.append(sl_loss.detach().cpu().numpy())
            if ema_model is not None:
                ssl_metrics.append(ssl_loss.detach().cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        confis = np.concatenate(confis, axis=0)
        labels = np.concatenate(labels, axis=0)
        noisy_labels = np.concatenate(noisy_labels, axis=0)
        self_pseudo_labels = np.concatenate(self_pseudo_labels, axis=0)
        topk_preds_list = np.concatenate(topk_preds_list, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        entropies = np.concatenate(entropies, axis=0)
        sl_losses = np.concatenate(sl_losses)
        if ema_model is not None:
            ssl_metrics = np.concatenate(ssl_metrics)
        lid = lid_term(feats, feats, k=20)
        model.train()

        # knn classifier
        neigh = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=12)
        neigh.fit(feats, noisy_labels)
        preds = neigh.predict(feats)
        acc = accuracy_score(labels, preds)
        client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{client_trainer.g_cid} n KNN: {acc*100:.2f}%")

        topk_knn_preds = []
        feat_cdist = cdist(feats, feats, metric="cosine") # 1-cos(a,b)
        # get k largest neighbors
        topk_idx = np.argsort(feat_cdist, axis=1)[:, :5]
        for i in range(feats.shape[0]):
            topk_labels = topk_preds_list[topk_idx[i]]
            # topk_labels = noisy_labels[topk_idx[i]]
            topk_labels = topk_labels.flatten()
            topk_labels = np.bincount(topk_labels, minlength=CLASS_NUM[client_trainer.args.dataset]).argmax()
            topk_knn_preds.append(topk_labels)
        topk_knn_preds = np.array(topk_knn_preds)
        topk_acc = accuracy_score(labels, topk_knn_preds)
        client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{client_trainer.g_cid} TopK KNN: {topk_acc*100:.2f}%")

        metric = sl_losses 

        return metric, clean_mask
    
    def on_client_training_start(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=16) 
        model = client_trainer.model
        # model = client_trainer.local_ema_models[client_trainer.l_cid]
        if client_trainer.args.use_orchestra:
            ema_model = client_trainer.local_ema_models[client_trainer.l_cid]
        else:
            ema_model = None
        metric,clean_mask = self.process(client_trainer, model, dataloader, ema_model=ema_model)
    #     df = pd.DataFrame({"metric": metric, "clean_mask": clean_mask})
    #     self.df_list.append(df)
    
    # def on_local_process_end(self, client_trainer, *args, **kwargs):
    #     df = pd.concat(self.df_list)
    #     # plot the loss distribution according to clean_mask
    #     plt.figure()
    #     sns.histplot(df, x="metric", hue="clean_mask", kde=True)
    #     plt.savefig(f"metric_distribution_{client_trainer.round:03}.png")
    #     plt.close()
    #     self.df_list = []
        
    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if client_trainer.round != 0:
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})
        else:
            self.sample_probs = {}
