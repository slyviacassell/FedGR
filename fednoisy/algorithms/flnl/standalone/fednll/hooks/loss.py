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
import scipy
from sklearn.utils.extmath import weighted_mode

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

class SLSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.ssl_weight = client_trainer.args.ssl_weight
        self.sl_weight = client_trainer.args.sl_weight
        self.ssl2sl_reg_weight = client_trainer.args.ssl2sl_reg_weight

    def sl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        return TF.cross_entropy(outputs["linear_head"]["cls_head"], targets["linear_head"]) * self.sl_weight
    
    def ssl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        raise NotImplementedError
    
    def ssl2sl_rep_reg(self, outputs):
        raise NotImplementedError
    
    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        sl_loss = self.sl_loss(client_trainer, outputs, targets, *args, **kwargs)
        ssl_loss = self.ssl_loss(client_trainer, outputs, targets, *args, **kwargs)
        loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs) 
        return loss


class SemiSLSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "cid_n_samples"):
            self.cid_n_samples = [0] * len(state['est_cid_noise'])

    def on_init(self, client_trainer, *args, **kwargs):
        # client_trainer.max_confis_ema = {cid: torch.ones(1, device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset] for cid in range(client_trainer.num_clients)}
        # client_trainer.confis_ema = {cid: (torch.ones(CLASS_NUM[client_trainer.args.dataset], device=client_trainer.device)/CLASS_NUM[client_trainer.args.dataset]) for cid in range(client_trainer.num_clients)}
        client_trainer.max_confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        client_trainer.confis_ema = {cid: None for cid in range(client_trainer.num_clients)}
        self.confi_gamma = client_trainer.args.confi_gamma
   
        self.local_epoch_cnt = 0

        self.relabels = {cid: {} for cid in range(client_trainer.num_clients)}
        self.sample_probs = {}
        self.est_cid_noise = [0] * client_trainer.args.num_clients
        self.cid_n_samples = [0] * client_trainer.args.num_clients

        self.pred_history = {cid: {} for cid in range(client_trainer.num_clients)}

    def on_client_training_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} "
            f"Client-{client_trainer.l_cid} "
            f"max confi: {client_trainer.max_confis_ema[client_trainer.l_cid].item():.4f} "
            f"confi: {client_trainer.confis_ema[client_trainer.l_cid].cpu().numpy()}"
        )

        self.local_epoch_cnt = 0

    def on_training_epoch_start(self, client_trainer, *args, **kwargs):
        if client_trainer.round >= client_trainer.args.sniffing_round:
            if self.local_epoch_cnt == 0: # global model
                cid = client_trainer.l_cid
                data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=128)
                guids, pseudo_labels = self.online_pseudo_labeling(client_trainer, data_loader, cid)
                # todo: maintain the pseudo labels
                self.relabels[cid].update({g: l for g, l in zip(guids, pseudo_labels)})
                for g, p in zip(guids, pseudo_labels):
                    if g not in self.pred_history[cid]:
                        self.pred_history[cid][g] = [p]
                    else:
                        self.pred_history[cid][g].append(p)

    def on_training_epoch_end(self, client_trainer, *args, **kwargs):
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{client_trainer.g_cid} p_acc: {self.p_meter.avg*100:.2f}%"
        )
        self.p_meter.reset()

        self.local_epoch_cnt += 1
        if client_trainer.round >= client_trainer.args.sniffing_round:
            # if self.local_epoch_cnt == 1:
            cid = client_trainer.l_cid
            # data_loader = client_trainer.dataset.get_eval_train_dataloader(client_trainer.args.dataset,cid=cid, batch_size=128)
            data_loader = client_trainer.dataset.get_semiws_dataloader(cid=cid, train=True, batch_size=128)
            # data_loader = client_trainer.dataset.get_dividemix_dataloader(cid=cid, train=True, batch_size=128)
            guids, pseudo_labels = self.online_pseudo_labeling(client_trainer, data_loader, cid)
            # self.relabels.update({g: l for g, l in zip(guids, pseudo_labels)})

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.get_est_cid_noise(client_trainer)

        if client_trainer.round >= client_trainer.args.sniffing_round:
            self.sample_probs.update({g: c for g,c in zip(client_trainer.overall_guids.tolist(), client_trainer.overall_probs.tolist())})

    @torch.no_grad()
    def online_pseudo_labeling(self, client_trainer, dataloader, cid, *args, **kwargs):
        gmm_p_meter = AverageMeter()
        p_meter = AverageMeter()

        model = client_trainer.model
        model.eval()

        guids_list = []
        p_targets_list = []

        for batch in dataloader:
            # imgs_w, guids, noisy_targets, targets = batch["img"], batch["guid"], batch["noisy_label"], batch["label"]
            imgs_w, guids, noisy_targets, targets, imgs_s = batch["img_w"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]
            # imgs_w, imgs_s, guids, noisy_targets, targets = batch["img_0"], batch["img_1"], batch["guid"], batch["noisy_label"], batch["label"], batch["img_s"]

            if client_trainer.cuda:
                imgs_w = imgs_w.to(client_trainer.device)
                # imgs_s = imgs_s.to(client_trainer.device)
                noisy_targets = noisy_targets.to(client_trainer.device)
                targets = targets.to(client_trainer.device)

            outputs_w = model(imgs_w, return_dict=True, full_heads=True)
            # outputs_s = model(imgs_s, return_dict=True, full_heads=True)
            logits = outputs_w["linear_head"]["cls_head"]
            confis = torch.softmax(logits, dim=1)
            probs, preds = torch.max(confis, dim=1)

            max_confi_ema = torch.clamp(client_trainer.max_confis_ema[cid], 0.5, 0.9)
            confi_ema = client_trainer.confis_ema[cid]
            # max norm
            confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            # mask = (probs > torch.clamp(confi_threshold[preds],0.5, 0.9))
            mask = (probs > 0.9)

            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids.numpy()]).to(client_trainer.device)
            mask = (mask & noisy_mask)

            noisy_targets[mask] = preds[mask]
            gmm_p_meter.update(torch.mean((targets[noisy_mask] == noisy_targets[noisy_mask]).float()).item(), len(targets[noisy_mask])+1e-8)
            p_meter.update(torch.mean((targets[mask] == noisy_targets[mask]).float()).item(), len(targets[mask])+1e-8)

            # hard label
            mask = (mask & noisy_mask) | (~noisy_mask)
            p_targets_list += noisy_targets[mask].cpu().tolist() 
            guids_list += guids[mask].tolist()

            # global model confi ema
            confi_gamma = 0.9
            if client_trainer.max_confis_ema[client_trainer.l_cid] is None:
                client_trainer.max_confis_ema[client_trainer.l_cid] = torch.max(confis, dim=1)[0].mean()
            else:
                client_trainer.max_confis_ema[client_trainer.l_cid] = confi_gamma * client_trainer.max_confis_ema[client_trainer.l_cid] + (1 - confi_gamma) * torch.max(confis, dim=1)[0].mean()
        
            if client_trainer.confis_ema[client_trainer.l_cid] is None:
                client_trainer.confis_ema[client_trainer.l_cid] = torch.mean(confis, dim=0)
            else:
                client_trainer.confis_ema[client_trainer.l_cid] = confi_gamma * client_trainer.confis_ema[client_trainer.l_cid] + (1 - confi_gamma) * torch.mean(confis, dim=0)

        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} client-{cid} gmm pseudo acc: {gmm_p_meter.avg*100:.2f}%, cnt: {gmm_p_meter.sum:.0f}/{gmm_p_meter.count:.0f}, "
            f"pseudo acc: {p_meter.avg*100:.2f}%, cnt: {p_meter.sum:.0f}/{p_meter.count:.0f}"
        )

        model.train()

        return guids_list, p_targets_list

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            self.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

            self.cid_n_samples[cid] = len(n_mask)

        sup_weights = [len(self.relabels[cid])/self.cid_n_samples[cid] for cid in client_trainer.id_list]
        sup_weights = np.array(sup_weights)
        mask = sup_weights < 0.2
        sup_weights[mask] = 0.0
        client_trainer._LOGGER.info(
            f"Round {client_trainer.round} "
            f"sup weights: {sup_weights}"
        )
    
    @torch.no_grad()
    def confidence_ema(self, client_trainer, confis: torch.Tensor):
        if client_trainer.max_confis_ema[client_trainer.l_cid] is None:
            client_trainer.max_confis_ema[client_trainer.l_cid] = torch.max(confis, dim=1)[0].mean()
        else:
            client_trainer.max_confis_ema[client_trainer.l_cid] = self.confi_gamma * client_trainer.max_confis_ema[client_trainer.l_cid] + (1 - self.confi_gamma) * torch.max(confis, dim=1)[0].mean()
    
        if client_trainer.confis_ema[client_trainer.l_cid] is None:
            client_trainer.confis_ema[client_trainer.l_cid] = torch.mean(confis, dim=0)
        else:
            client_trainer.confis_ema[client_trainer.l_cid] = self.confi_gamma * client_trainer.confis_ema[client_trainer.l_cid] + (1 - self.confi_gamma) * torch.mean(confis, dim=0)


class SupOrchestraLoss(SLSSLLoss):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.orchestra_temperature = client_trainer.args.orchestra_temperature
        SLSSLLoss.on_init(self, client_trainer, *args, **kwargs)

    def loss(self, client_trainer, outputs_s, targets, *args, **kwargs):
        sl_loss = self.sl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
        ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
        loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs_s) 
        return loss
    
    def ssl_loss(self, client_trainer, outputs, targets, *args, **kwargs):
        return -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs["orchestra_head"]+1e-10, dim=1), dim=1).mean() * self.ssl_weight
    
    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["linear_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["orchestra_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer * self.ssl2sl_reg_weight
    

class SemiOrchestraLoss(SupOrchestraLoss, SemiSLSSLLoss):
    def __init__(self) -> None:
        super().__init__()
        self.p_meter = AverageMeter()

    def on_init(self, client_trainer, *args, **kwargs):
        SupOrchestraLoss.on_init(self, client_trainer, *args, **kwargs)
        SemiSLSSLLoss.on_init(self, client_trainer, *args, **kwargs)

    def loss(self, client_trainer, outputs_w, outputs_s, targets, *args, **kwargs):
        with torch.no_grad():
            confis = torch.softmax(outputs_w["linear_head"]["cls_head"], dim=1)
            self.confidence_ema(client_trainer, confis)
        
        if client_trainer.round < client_trainer.args.sniffing_round:
            loss = super().loss(client_trainer, outputs_s, targets, *args, **kwargs)
            return loss
        elif client_trainer.round < client_trainer.args.warmup_round + client_trainer.args.sniffing_round:
            # linear decay according to the noise level and the round
            sl_weight = (1. - self.est_cid_noise[client_trainer.l_cid]) * min(1., (client_trainer.round - client_trainer.args.sniffing_round) / min(client_trainer.args.warmup_round, 50))
            sl_loss = self.sl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
            sl_loss = sl_loss * sl_weight
        
            ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)

            loss = sl_loss + ssl_loss + self.ssl2sl_rep_reg(outputs_s)
            return loss
        else:
            guids = kwargs["guids"].numpy().tolist()
            labels = kwargs["labels"]
            
            sup_targets = targets["linear_head"]

            self.p_meter.update(torch.mean((labels == sup_targets.cpu()).float()).item(), len(labels))

            noisy_mask = torch.tensor([True if g in client_trainer.overall_noisy_guids else False for g in guids]).to(client_trainer.device)
            noisy_targets = TF.one_hot(sup_targets, num_classes=CLASS_NUM[client_trainer.args.dataset]).float()
            n_probs = torch.tensor([self.sample_probs.get(g, 1) for g in guids], device=client_trainer.device, dtype=torch.float32).unsqueeze(1)
            
            # hard label
            cid = client_trainer.l_cid
            p_mask = torch.tensor([True if g in self.relabels[cid] else False for g in guids]).to(client_trainer.device)
            pseudo_targets = [self.relabels[cid].get(g, sup_targets[i]) for i, g in enumerate(guids)] 
            pseudo_targets = TF.one_hot(torch.tensor(pseudo_targets).to(client_trainer.device), num_classes=CLASS_NUM[client_trainer.args.dataset]).float()

            pseudo_targets[~noisy_mask] = noisy_targets[~noisy_mask] * n_probs[~noisy_mask] + (1. - n_probs[~noisy_mask]) * pseudo_targets[~noisy_mask]

            logits = outputs_s["linear_head"]["cls_head"]
            probs, preds = torch.max(torch.softmax(logits, dim=1), dim=1)
            
            # max_confi_ema = torch.clamp(client_trainer.max_confis_ema[cid], 0.5, 0.9)
            # confi_ema = client_trainer.confis_ema[client_trainer.l_cid]
            # # max norm
            # confi_threshold = confi_ema/confi_ema.max() * max_confi_ema
            # mask = (probs > confi_threshold[preds])
            # mask = (mask & noisy_mask) | (~noisy_mask)
            mask = p_mask

            sl_loss = -torch.sum(pseudo_targets * torch.log_softmax(logits+1e-10, dim=1), dim=1)
            if mask.sum() > 0:
                sl_loss = sl_loss[mask].mean()
            else:
                sl_loss = 0.
            sup_weights = len(self.relabels[cid])/self.cid_n_samples[cid]
            if sup_weights < 0.2:
                sup_weights = 0.0

            ssl_loss = self.ssl_loss(client_trainer, outputs_s, targets, *args, **kwargs)
            
            loss = sl_loss * sup_weights + ssl_loss + self.ssl2sl_rep_reg(outputs_s)
            return loss


# === SSL Losses ===
class SimpleSSLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.temperature = client_trainer.args.orchestra_temperature  

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        ssl_loss = -torch.sum(torch.softmax(targets["simplessl_head"] / self.temperature, dim=-1) * torch.log_softmax(outputs["simplessl_head"] / self.temperature + 1e-10, dim=1), dim=1).mean()
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs)
        return ssl_loss, ssl2sl_reg

    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["cls_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["simplessl_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer    


class OrchestraLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
    
    def on_init(self, client_trainer, *args, **kwargs):
        self.orchestra_temperature = client_trainer.args.orchestra_temperature

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        ssl_loss = -torch.sum(targets["orchestra_head"] * torch.log_softmax(outputs["orchestra_head"] / self.orchestra_temperature + 1e-10, dim=1), dim=1).mean()
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs)
        return ssl_loss, ssl2sl_reg
    
    def ssl2sl_rep_reg(self, outputs):
        sl_normlized = TF.normalize(outputs["cls_head"]["cls_embedding"], p=2, dim=1)
        ssl_normlized = TF.normalize(outputs["orchestra_head"], p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer
    

class SimSiamLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, client_trainer, outputs, targets=None, *args, **kwargs):
        z1, z2, p1, p2 = outputs["projector_predictor"]["z"][0], outputs["projector_predictor"]["z"][1], outputs["projector_predictor"]["p"][0], outputs["projector_predictor"]["p"][1]
        ssl_loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][0], outputs["projector_predictor"]["z"][0]) / 2 + self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][1], outputs["projector_predictor"]["z"][1]) / 2
        return ssl_loss, ssl2sl_reg
    
    def ssl2sl_rep_reg(self, sl_embeddings, ssl_embeddings):
        sl_normlized = TF.normalize(sl_embeddings, p=2, dim=1)
        ssl_normlized = TF.normalize(ssl_embeddings, p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer

    def D(self, p, z):  # negative cosine similarity
        # return - TF.cosine_similarity(p, z.detach(), dim=-1).mean()
        return (1. - TF.cosine_similarity(p, z.detach(), dim=-1)).mean()


class BYOLLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()
    
    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        z1, z2, p1, p2 = outputs["projector_predictor"]["z"][0], outputs["projector_predictor"]["z"][1], outputs["projector_predictor"]["p"][0], outputs["projector_predictor"]["p"][1]
        ema_z1, ema_z2 = targets["projector_predictor"]["z"][0], targets["projector_predictor"]["z"][1]
        
        ssl_loss = self.byol_loss_fn(p1, ema_z2) / 2 + self.byol_loss_fn(p2, ema_z1) / 2
        ssl_loss = ssl_loss.mean()
        ssl2sl_reg = self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][0], outputs["projector_predictor"]["z"][0]) / 2 + self.ssl2sl_rep_reg(outputs["cls_head"]["cls_embedding"][1], outputs["projector_predictor"]["z"][1]) / 2
        return ssl_loss, ssl2sl_reg
        
    def byol_loss_fn(self, x, y):
        # x = TF.normalize(x, dim=-1, p=2)
        # y = TF.normalize(y, dim=-1, p=2)
        # return 2 - 2 * (x * y).sum(dim=-1)
        return (2. - 2 * TF.cosine_similarity(x, y.detach(), dim=-1)).mean()

    
    def ssl2sl_rep_reg(self, sl_embeddings, ssl_embeddings):
        sl_normlized = TF.normalize(sl_embeddings, p=2, dim=1)
        ssl_normlized = TF.normalize(ssl_embeddings, p=2, dim=1)
        sl_gram = torch.mm(sl_normlized, sl_normlized.t())
        ssl_gram = torch.mm(ssl_normlized, ssl_normlized.t())
        regulizer = torch.sum((sl_gram - ssl_gram) ** 2,dim=1).mean()
        return regulizer
# === SSL Losses ===
  
    
class SemiSupLoss(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        self.est_cid_noise = [0] * client_trainer.args.num_clients
        self.cid_n_samples = [0] * client_trainer.args.num_clients

    def on_local_process_start(self, client_trainer, *args, **kwargs):
        self.get_est_cid_noise(client_trainer)

    def get_est_cid_noise(self, client_trainer, *args, **kwargs):
        for cid in client_trainer.id_list:
            dataset = client_trainer.dataset.get_dataset(cid=cid, train=True)
            n_mask = [True if g in client_trainer.overall_noisy_guids else False for g in dataset.guids]
            self.est_cid_noise[cid] = sum(n_mask)/len(n_mask)

            self.cid_n_samples[cid] = len(n_mask)

    def loss(self, client_trainer, outputs, targets, *args, **kwargs):
        if client_trainer.round < client_trainer.args.sniffing_round:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])
            return loss
        elif client_trainer.round < client_trainer.args.warmup_round + client_trainer.args.sniffing_round:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])
            return loss
        else:
            loss = TF.cross_entropy(outputs["cls_head"]["cls_logits"], targets["cls_head"])
            return loss