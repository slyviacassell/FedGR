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
from tqdm import tqdm
from sklearn.manifold import TSNE

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)


class EmbeddingVisualize(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def visulize(self, client_trainer, *args, **kwargs):
        pass


class OrchestraEmbeddingTSNE(EmbeddingVisualize):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def visulize(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_overall_dataloader()
        model = client_trainer.model
        model.eval()
        feats = []
        labels = []
        clean_mask = []
        for batch in tqdm(dataloader, desc="TSNE Embedding"):
            imgs, targets, noisy_targets = batch["img"], batch["label"], batch["noisy_label"]
            imgs = imgs.to(client_trainer.device)
            feat = model(imgs, return_dict=True, full_heads=True)["orchestra_head"]
            feat = TF.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            clean_mask.append(noisy_targets.numpy() == targets.numpy())
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        model.train()

        global_centroids = client_trainer.global_centroids.weight.data.cpu().numpy()

        # add the global centroids to the feature matrix
        feats = np.concatenate([feats, global_centroids], axis=0)
        labels = np.concatenate([labels, np.arange(global_centroids.shape[0])], axis=0)
        clean_mask = np.concatenate([clean_mask, -1 * np.ones(global_centroids.shape[0], dtype=bool)], axis=0)

        # TSNE
        tsne = TSNE(n_components=2, random_state=client_trainer.args.seed, n_jobs=12, metric="cosine")
        feats = tsne.fit_transform(feats)
        
        df = pd.DataFrame(feats, columns=["x", "y"])
        df["label"] = labels
        df["clean"] = clean_mask
        
        # Plot feat tsne
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df[df["clean"] != -1], x="x", y="y", hue="label", style="clean", palette="tab10", s=10)

        # Plot centroids tsne
        centroids = df[df["clean"] == -1]
        for i in range(centroids.shape[0]):
            plt.scatter(centroids.iloc[i]["x"], centroids.iloc[i]["y"], s=12, c="black", marker="*")
            # plt.text(centroids.iloc[i]["x"], centroids.iloc[i]["y"] + 1, str(int(centroids.iloc[i]["label"])), fontsize=12, color="black")

        plt.title(f"Orchestra TSNE Embedding")
        plt.savefig(f"tmp/tsne_{client_trainer.round}.png", dpi=300)
        plt.close()
        

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if self.every_n_round(client_trainer, 10):
            self.visulize(client_trainer, *args, **kwargs)


class BackboneEmbeddingTSNE(EmbeddingVisualize):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def visulize(self, client_trainer, *args, **kwargs):
        dataloader = client_trainer.dataset.get_overall_dataloader()
        model = client_trainer.model
        model.eval()
        feats = []
        labels = []
        clean_mask = []
        for batch in tqdm(dataloader, desc="TSNE Embedding"):
            imgs, targets, noisy_targets = batch["img"], batch["label"], batch["noisy_label"]
            imgs = imgs.to(client_trainer.device)
            feat = model.get_embedding(imgs)
            feat = TF.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            clean_mask.append(noisy_targets.numpy() == targets.numpy())
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        clean_mask = np.concatenate(clean_mask, axis=0)
        model.train()

        # TSNE
        tsne = TSNE(n_components=2, random_state=client_trainer.args.seed, n_jobs=12, metric="cosine")
        feats = tsne.fit_transform(feats)
        
        df = pd.DataFrame(feats, columns=["x", "y"])
        df["label"] = labels
        df["clean"] = clean_mask
        
        # Plot feat tsne
        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=df[df["clean"] != -1], x="x", y="y", hue="label", style="clean", palette="tab10", s=10)

        plt.title(f"Backbone TSNE Embedding")
        plt.savefig(f"tmp/tsne_{client_trainer.round}.png", dpi=300)
        plt.close()
        

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if self.every_n_round(client_trainer, 10):
            self.visulize(client_trainer, *args, **kwargs)
