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

from fednoisy.utils.misc import (
    AverageMeter,
)

from fednoisy.core.hooks import (
    SerialClientTrainerHook,
    SyncServerHook,
)


class NaivePseudoLabelLoss(SerialClientTrainerHook):
    def __init__(self, num_clients) -> None:
        super().__init__()
        self.client_pseudo_labels = [{} for _ in range(num_clients)]
        self.pseudo_label_monitors = [AverageMeter() for _ in range(num_clients)]

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        data_loader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=client_trainer.g_cid, batch_size=128)
        p_model = client_trainer.cur_global_model
        p_model.eval()
        self.client_pseudo_labels[client_trainer.l_cid] = {}
        # using the p_model to pseudo label the data and store the pseudo labels wiht (guid, pseudo_label) format
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels, noisy_labels = batch["img"], batch["label"], batch["noisy_label"]
                guids = batch["guid"]
                inputs = inputs.to(client_trainer.device)
                outputs = p_model(inputs)
                outputs = TF.softmax(outputs, dim=1)
                max_prob, max_idx = torch.max(outputs, dim=1)
                mask = max_prob > 0.95
                for i in range(len(guids)):
                    if mask[i]:
                        self.client_pseudo_labels[client_trainer.l_cid][guids[i].item()] = max_idx[i].item()

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        assert client_trainer.args.warmup_round > 1
        if client_trainer.round < client_trainer.args.warmup_round:
            loss = client_trainer.criterion(outputs, targets)
        else:
            guids = kwargs["guids"]
            mask = [False if g in client_trainer.overall_noisy_guids else True for g in guids.numpy()] # mask out noisy samples
            mask = torch.tensor(mask).to(client_trainer.device)
            bsz = outputs.size(0)
            
            if torch.sum(mask) != 0:
                sup_loss = TF.cross_entropy(outputs, targets,reduction="none")
                sup_loss = (sup_loss * mask).mean()
            else:
                sup_loss = 0.

            pseudo_mask = (~mask)
            pseudo_labels = torch.tensor([self.client_pseudo_labels[client_trainer.l_cid][g.item()] if g.item() in self.client_pseudo_labels[client_trainer.l_cid] else -1 for g in guids], dtype=torch.long)
            pseudo_mask = (pseudo_labels != -1) & pseudo_mask.cpu()
            if torch.sum(pseudo_mask) != 0:
                # replace the noisy labels with pseudo labels if the guid is in the client_pseudo_labels

                gt = kwargs["gt"]
                self.pseudo_label_monitors[client_trainer.l_cid].update(torch.sum(pseudo_labels[pseudo_mask] == gt[pseudo_mask]).item() / torch.sum(pseudo_mask).item() + 1e-8, torch.sum(pseudo_mask).item() + 1e-8)

                pseudo_labels = pseudo_labels.to(client_trainer.device)
                pseudo_mask = pseudo_mask.to(client_trainer.device)
                pseudo_loss = TF.cross_entropy(outputs[pseudo_mask], pseudo_labels[pseudo_mask], reduction="none")
                pseudo_loss = pseudo_loss.sum() / bsz
            else:
                pseudo_loss = 0.
            
            loss = sup_loss + pseudo_loss
                
        return loss
    
    def on_client_training_end(self, client_trainer, *args, **kwargs):
        if client_trainer.round >= client_trainer.args.warmup_round:
            client_trainer._LOGGER.info(
                f"Round {client_trainer.round} client-{client_trainer.g_cid} pseudo label acc: {self.pseudo_label_monitors[client_trainer.l_cid].avg*100:.2f}%, pseudo label count: {self.pseudo_label_monitors[client_trainer.l_cid].count:.2f}"
            )

            self.pseudo_label_monitors[client_trainer.l_cid].reset()
