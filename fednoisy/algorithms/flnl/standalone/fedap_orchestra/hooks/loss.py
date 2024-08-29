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


class SupOrchestraLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs_s, targets, *args, **kwargs):
        sup_loss = TF.cross_entropy(outputs_s["linear_head"], targets["linear_head"])
        
        orchestra_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs_s["orchestra_head"]+1e-10, dim=1), dim=1).mean()

        loss = sup_loss + orchestra_loss
        return loss
    

class SemiOrchestraLoss(SupOrchestraLoss):
    def __init__(self) -> None:
        super().__init__()
        self.p_meter = AverageMeter()

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.max_confis_ema = {cid: torch.ones(1, device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset] for cid in range(client_trainer.num_clients)}
        client_trainer.confis_ema = {cid: (torch.ones(CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset]) for cid in range(client_trainer.num_clients)}
        self.confi_beta = 0.999

        self.pseudo_labels = {}
        self.sample_probs = {}

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} "
            f"Client-{client_trainer.l_cid} "
            f"max confi: {client_trainer.max_confis_ema[client_trainer.l_cid].item():.4f} "
            f"confi: {client_trainer.confis_ema[client_trainer.l_cid].cpu().numpy()}"
        )

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{client_trainer.g_cid} p_acc: {self.p_meter.avg*100:.2f}%"
        )
        self.p_meter.reset()

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        if client_trainer.round >= client_trainer.args.warmup_round:
            for cid in client_trainer.id_list:
                dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
                data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=128)
                guids, pseudo_labels = self.pseudo_labeling(client_trainer, data_loader, cid)
                # for i,g in enumerate(dataset.guids):
                #     dataset.noisy_labels[i] = pseudo_labels[guids.index(g)]
                # client_trainer._LOGGER.info(f"Round {client_trainer.round} client-{cid} pseudo acc: {accuracy_score(dataset.labels, dataset.noisy_labels)*100:.2f}%")
                
                self.pseudo_labels.update({g: l for g, l in zip(guids, pseudo_labels)})
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})

    @torch.no_grad()
    def pseudo_labeling(self, client_trainer, dataloader, cid, *args, **kwargs):
        client_trainer.model.eval()
        guids_list = []
        targets_list = []
        for batch in dataloader:
            imgs, guids, targets = batch["img_w"], batch["guid"], batch["noisy_label"]
            if client_trainer.cuda:
                imgs = imgs.to(client_trainer.device)
                targets = targets.to(client_trainer.device)

            outputs = client_trainer.model(imgs, return_dict=True, full_heads=True)
            probs, preds = torch.max(torch.softmax(outputs["linear_head"], dim=1), dim=1)

            max_confi_ema = client_trainer.max_confis_ema[cid]
            confi_ema = client_trainer.confis_ema[cid]
            # max norm
            confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            mask = (probs > confi_threshold[preds])

            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy()]).to(client_trainer.device)
            mask = mask & noisy_mask

            targets[mask] = preds[mask]

            guids_list += guids.tolist()
            targets_list += targets.cpu().tolist()
        return guids_list, targets_list

    def loss(self, client_trainer, outputs_w, outputs_s, targets, *args, **kwargs):
        with torch.no_grad():
            confis = torch.softmax(outputs_w["linear_head"], dim=1)
            self.confidence_ema(client_trainer, confis)

        if client_trainer.round < client_trainer.args.warmup_round:
            return super().loss(client_trainer, outputs_s, targets, *args, **kwargs)
        else:
            labels = kwargs["labels"]
            guids = kwargs["guids"].numpy().tolist()
            
            sup_targets = targets["linear_head"]

            self.p_meter.update(torch.mean((labels == sup_targets.cpu()).float()).item(), len(labels))

            # sup_loss = TF.cross_entropy(outputs_s["linear_head"], sup_targets)


            noisy_targets = TF.one_hot(sup_targets, num_classes=CLASS_NUM[client_trainer.args.dataset]).float()
            probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids], device=client_trainer.device, dtype=torch.float32).unsqueeze(1)
            pseudo_targets = [self.pseudo_labels.get(g, sup_targets[i]) for i,g in enumerate(guids)]
            pseudo_targets = TF.one_hot(torch.tensor(pseudo_targets).to(client_trainer.device), num_classes=CLASS_NUM[client_trainer.args.dataset]).float()
            sup_targets = probs * noisy_targets + (1. - probs) * pseudo_targets
            
            sup_loss = -torch.sum(sup_targets * torch.log_softmax(outputs_s["linear_head"]+1e-10, dim=1), dim=1).mean()


            orchestra_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs_s["orchestra_head"]+1e-10, dim=1), dim=1).mean()
            loss = sup_loss + orchestra_loss
            return loss
        
    @torch.no_grad()
    def confidence_ema(self, client_trainer, confis: torch.Tensor):
        client_trainer.max_confis_ema[client_trainer.l_cid] = self.confi_beta * client_trainer.max_confis_ema[client_trainer.l_cid] + (1 - self.confi_beta) * torch.max(confis, dim=1)[0].mean()
        client_trainer.confis_ema[client_trainer.l_cid] = self.confi_beta * client_trainer.confis_ema[client_trainer.l_cid] + (1 - self.confi_beta) * torch.mean(confis, dim=0)