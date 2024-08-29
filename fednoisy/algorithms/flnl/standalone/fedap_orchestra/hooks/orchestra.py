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
    SyncServerHook,
)
from fednoisy.utils.misc import (
    AverageMeter,
)
from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)


class GlobalOrchestra(SyncServerHook):
    def __init__(self) -> None:
        super().__init__()
    
    def on_init(self, handler, *args, **kwargs):
        self.feat_dim = handler.args.feat_dim
        self.g_n_centroids = handler.args.g_n_centroids
        self.orchestra_temperature = handler.args.orchestra_temperature

        self.centroids = nn.Linear(self.feat_dim, self.g_n_centroids, bias=False)

    def on_global_update_start(self, handler, *args, **kwargs):
        local_centroids = torch.cat(handler.local_centroids, dim=0) # [N_centroids * num_clients, D]
        self.clustering(local_centroids)
    
    def get_centroids(self, handler, *args, **kwargs):
        return self.centroids.weight.data.clone()
    
    def clustering(self, local_centroids: torch.Tensor):
        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        train_loss = 0.
        total_rounds = 500
        self.centroids.train()

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(TF.normalize(local_centroids, dim=1)))

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = TF.softmax(self.centroids(TF.normalize(local_centroids, dim=1)) / self.orchestra_temperature, dim=1) # optimal assignments refer to Genevay et al.
            # Match predicted assignments with SK assignments
            loss = (1 - TF.cosine_similarity(SK_assigns, probs1, dim=-1)).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.centroids.weight.copy_(TF.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()
            if (round_idx + 1) % 100 == 0:
                print(f"Round {round_idx + 1}/{total_rounds} Loss: {train_loss / (round_idx + 1):.4f}")


class LocalOrchestra(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.feat_dim = client_trainer.args.feat_dim
        self.l_n_centroids = client_trainer.args.l_n_centroids
        self.g_n_centroids = client_trainer.args.g_n_centroids
        self.queue_size = client_trainer.args.queue_size
        self.orchestra_temperature = client_trainer.args.orchestra_temperature

        client_trainer.global_centroids = nn.Linear(self.feat_dim, self.g_n_centroids, bias=False)
        client_trainer.global_centroids.to(client_trainer.device)
        client_trainer.global_centroids.eval()

        client_trainer.queues = {cid: torch.randn(self.feat_dim, self.queue_size) for cid in range(client_trainer.num_clients)}
        client_trainer.queue_ptr = {cid: 0 for cid in range(client_trainer.num_clients)}

        # debug
        client_trainer.queue_labels = {cid: torch.zeros(self.queue_size, dtype=torch.long) for cid in range(client_trainer.num_clients)}
        client_trainer.queue_guids = {cid: torch.zeros(self.queue_size, dtype=torch.long) for cid in range(client_trainer.num_clients)}

        client_trainer.local_centroids = []

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        global_centroids = client_trainer.cur_payload[6]
        client_trainer.global_centroids.weight.data.copy_(global_centroids)

    @torch.no_grad()
    def get_assignment(self, client_trainer, outputs_w: torch.Tensor, outputs_s: torch.Tensor, *args, **kwargs):
        keys = TF.normalize(outputs_w["orchestra_head"], dim=1)
        
        self.dequeue_and_enqueue(client_trainer, keys.cpu(), kwargs.get("labels", None), kwargs.get("guids", None))
        
        q = TF.softmax(client_trainer.global_centroids(keys) / self.orchestra_temperature, dim=1)

        return q

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        local_centroids = self.clustering(client_trainer)
        client_trainer.local_centroids.append(local_centroids)

    def on_local_process_end(self, client_trainer, *args, **kwargs):
        client_trainer.local_centroids = []

    @torch.no_grad()
    def clustering(self, client_trainer):
        feats = client_trainer.queues[client_trainer.l_cid].T.clone() # [N, D]
        centroids = feats[np.random.choice(feats.shape[0], self.l_n_centroids, replace=False)].clone() # [N_centroids, D]
        local_iters = 5
        # sinkhorn + kmeans
        for it in range(local_iters):
            assigns = sknopp(TF.normalize(feats,dim=1) @ TF.normalize(centroids.T,dim=0), max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            for index in range(self.l_n_centroids):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected = torch.index_select(feats, 0, selected)
                if selected.shape[0] == 0:
                    selected = feats[torch.randint(len(feats), (1,))]
                centroids[index] = TF.normalize(selected.mean(dim=0), dim=0)
        
        return centroids # [N_centroids, D]

    @torch.no_grad()
    def dequeue_and_enqueue(self, client_trainer, keys: torch.Tensor, labels: torch.Tensor=None, guids: torch.Tensor=None):
        bsz = keys.shape[0]
        ptr = client_trainer.queue_ptr[client_trainer.l_cid]
        if ptr + bsz > self.queue_size:
            client_trainer.queues[client_trainer.l_cid][:, ptr:] = keys[:self.queue_size - ptr].T
            client_trainer.queues[client_trainer.l_cid][:, :bsz - (self.queue_size - ptr)] = keys[self.queue_size - ptr:].T

            if labels is not None:
                client_trainer.queue_labels[client_trainer.l_cid][ptr:] = labels[:self.queue_size - ptr]
                client_trainer.queue_labels[client_trainer.l_cid][:bsz - (self.queue_size - ptr)] = labels[self.queue_size - ptr:]
            if guids is not None:
                client_trainer.queue_guids[client_trainer.l_cid][ptr:] = guids[:self.queue_size - ptr]
                client_trainer.queue_guids[client_trainer.l_cid][:bsz - (self.queue_size - ptr)] = guids[self.queue_size - ptr:]
        else:
            client_trainer.queues[client_trainer.l_cid][:, ptr:ptr + bsz] = keys.T

            if labels is not None:
                client_trainer.queue_labels[client_trainer.l_cid][ptr:ptr + bsz] = labels
            if guids is not None:
                client_trainer.queue_guids[client_trainer.l_cid][ptr:ptr + bsz] = guids
        ptr = (ptr + bsz) % self.queue_size
        client_trainer.queue_ptr[client_trainer.l_cid] = ptr


# Sinkhorn Knopp 
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = TF.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations. 
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments